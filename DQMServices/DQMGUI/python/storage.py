import re
import time
import zlib
import struct
import socket
import asyncio
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

        if not row:
            # Blob doesn't exist, we should probably try to import
            return None

        me_list = await cls.__me_list_from_blob(row[0])
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
        me_list = await cls.__me_list_from_blob(row[1])
        me_infos = await cls.__me_infos_from_blob(row[2])

        return (filename, me_list, me_infos)


    @classmethod
    async def get_samples_count(cls):
        sql = 'SELECT COUNT(*) FROM samples;'
        cursor = await cls.__db.execute(sql)
        row = await cursor.fetchone()
        await cursor.close()
        return row[0]


    @classmethod
    async def get_sample_filename(cls, run, dataset):
        """
        Returns a full path to a ROOT file containing plots of a given sample. 
        Returns None if given sample doesn't exist.
        """

        sql = 'SELECT filename FROM samples WHERE run = ? AND dataset = ?;'
        cursor = await cls.__db.execute(sql, (int(run), dataset))
        row = await cursor.fetchone()

        await cursor.close()

        if row:
            return row[0]
        else:
            return None


    @classmethod
    async def add_samples(cls, samples):
        """
        Adds multiple samples to the database.
        Sample is declared here: data_types.SampleFull
        """

        sql = 'INSERT OR REPLACE INTO samples VALUES(?,?,0,?,NULL,NULL);'
        await cls.__db.executemany(sql, samples)
        await cls.__db.commit()


    @classmethod
    async def add_blobs(cls, me_list_blob, offsets_blob, run, dataset, filename):
        """Adds me list blob and offsets blob to a DB in a single transaction."""

        await cls.__db.execute('BEGIN;')

        await cls.__db.execute('INSERT OR IGNORE INTO menames (menameblob) VALUES (?);', (me_list_blob,))
        cursor = await cls.__db.execute('SELECT menamesid FROM menames WHERE menameblob = ?;', (me_list_blob,))
        row = await cursor.fetchone()
        me_list_blob_id = row[0]

        await cls.__db.execute('INSERT OR IGNORE INTO meoffsets (meoffsetsblob) VALUES (?);', (offsets_blob,))
        cursor = await cls.__db.execute('SELECT meoffsetsid FROM meoffsets WHERE meoffsetsblob = ?;', (offsets_blob,))
        row = await cursor.fetchone()
        offsets_blob_id = row[0]

        sql = 'UPDATE samples SET menamesid = ?, meoffsetsid = ? WHERE run = ? AND dataset = ? AND filename = ?;'
        await cls.__db.execute(sql, (me_list_blob_id, offsets_blob_id, int(run), dataset, filename))

        await cls.__db.execute('COMMIT;')


    @classmethod
    async def __me_list_from_blob(cls, namesblob):
        def me_list_from_blob_sync():
            return zlib.decompress(namesblob).splitlines()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, me_list_from_blob_sync)

    
    @classmethod
    async def __me_infos_from_blob(cls, offsetblob):
        def me_infos_from_blob_sync():
            return MEInfo.blobtolist(offsetblob)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, me_infos_from_blob_sync)

