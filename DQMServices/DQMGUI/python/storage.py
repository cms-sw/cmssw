import re
import time
import zlib
import struct
import socket
import tempfile
import aiosqlite
import subprocess

from meinfo import MEInfo


class GUIDataStore:

    __DBSCHEMA = """
    BEGIN;
    CREATE TABLE IF NOT EXISTS samples(dataset, run, lumi, filename, menamesid, meoffsetsid);
    CREATE INDEX IF NOT EXISTS samplelookup ON samples(dataset, run, lumi);
    CREATE UNIQUE INDEX IF NOT EXISTS uniquesamples on samples(dataset, run, lumi, filename);
    CREATE TABLE IF NOT EXISTS menames(menamesid INTEGER PRIMARY KEY, menameblob);
    CREATE UNIQUE INDEX IF NOT EXISTS menamesdedup ON menames(menameblob);
    CREATE TABLE IF NOT EXISTS meoffsets(meoffsetsid INTEGER PRIMARY KEY, meoffsetsblob);
    COMMIT;
    """

    __db = None

    @classmethod
    async def initialize(cls, connection_string='../data/directory.sqlite'):
        """Creates DB from schema if it doesn't exists and open a connection to it."""

        # TODO: Close connection at some point!
        cls.__db = await aiosqlite.connect(connection_string)
        await cls.__db.executescript(cls.__DBSCHEMA)


    @classmethod
    async def destroy(cls):
        try:
            await cls.__db.close()
        except:
            pass


    @classmethod
    async def get_samples(cls, run, dataset):
        if run:
            run = '%%%s%%' % run
        if dataset:
            dataset = '%%%s%%' % dataset

        cursor = None

        if run == dataset == None:
            return []
        elif run != None and dataset != None:
            sql = 'SELECT DISTINCT run, dataset FROM samples WHERE dataset LIKE ? AND run LIKE ?'
            cursor = await cls.__db.execute(sql, (dataset, run))
        elif run != None:
            sql = 'SELECT DISTINCT run, dataset FROM samples WHERE run LIKE ?'
            cursor = await cls.__db.execute(sql, (run,))
        elif dataset != None:
            sql = 'SELECT DISTINCT run, dataset FROM samples WHERE dataset LIKE ?'
            cursor = await cls.__db.execute(sql, (dataset,))

        rows = await cursor.fetchall()
        await cursor.close()

        return rows


    @classmethod
    async def get_me_list_blob(cls, run, dataset):
        """
        me_list format for an ME (ME/Path/mename) that has 2 QTests and an efficiency flag set looks like this:
        ME/Path/mename
        ME/Path/mename\0.qtest1
        ME/Path/mename\0.qtest2
        ME/Path/mename\0e=1
        This function returns a list of contents in provided run/dataset combination, in a format explained above.
        """
        # For now adding a LIMIT 1 because there might be multiple version of the same file.
        sql = 'SELECT menameblob FROM samples JOIN menames ON samples.menamesid = menames.menamesid WHERE run = ? AND dataset = ? LIMIT 1;'

        cursor = await cls.__db.execute(sql, (int(run), dataset))
        row = await cursor.fetchone()
        await cursor.close()

        me_list = cls.__me_list_from_blob(row[0])
        return me_list


    @classmethod
    async def get_blobs_and_filename(cls, run, dataset):
        # For now adding a LIMIT 1 because there might be multiple version of the same file.
        sql = '''
        SELECT
            filename,
            menameblob,
            meoffsetsblob
        FROM
            samples
            JOIN menames ON samples.menamesid = menames.menamesid
            JOIN meoffsets ON samples.meoffsetsid = meoffsets.meoffsetsid
        WHERE
            run = ?
        AND 
            dataset = ?
        LIMIT 1;
        '''

        cursor = await cls.__db.execute(sql, (int(run), dataset))
        row = await cursor.fetchone()
        await cursor.close()

        filename = row[0]
        me_list = cls.__me_list_from_blob(row[1])
        me_infos = cls.__me_infos_from_blob(row[2])

        return (filename, me_list, me_infos)


    @classmethod
    def __me_list_from_blob(cls, namesblob):
        return zlib.decompress(namesblob).splitlines()

    
    @classmethod
    def __me_infos_from_blob(cls, offsetblob):
        return MEInfo.blobtolist(offsetblob)

