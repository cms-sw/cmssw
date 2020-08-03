import re
import time
import zlib
import struct
import socket
import asyncio
import tempfile
import aiosqlite
import subprocess

from .helpers import get_absolute_path
from .compressing import GUIBlobCompressor


class GUIDataStore:

    __DBSCHEMA = """
    BEGIN;
    CREATE TABLE IF NOT EXISTS samples(dataset, run, lumi, filename, fileformat, menamesid, meinfosid, registered default (strftime('%s','now')));
    CREATE INDEX IF NOT EXISTS samplelookup ON samples(dataset, run, lumi);
    CREATE UNIQUE INDEX IF NOT EXISTS uniquesamples on samples(dataset, run, lumi);
    CREATE TABLE IF NOT EXISTS menames(menamesid INTEGER PRIMARY KEY, menameblob);
    CREATE UNIQUE INDEX IF NOT EXISTS menamesdedup ON menames(menameblob);
    CREATE TABLE IF NOT EXISTS meinfos(meinfosid INTEGER PRIMARY KEY, meinfosblob);
    COMMIT;
    """

    __db = None
    __lock = asyncio.Lock()

    compressor = GUIBlobCompressor()


    @classmethod
    async def initialize(cls, connection_string='../data/directory.sqlite', in_memory=False):
        """
        Creates DB from schema if it doesn't exists and open a connection to it.
        If in_memory is set to Trye, an in memory DB will be used.
        """

        if in_memory:
            connection_string = ':memory:'
        else:
            connection_string = get_absolute_path(connection_string)

        cls.__db = await aiosqlite.connect(connection_string)
        await cls.__db.executescript(cls.__DBSCHEMA)


    @classmethod
    async def destroy(cls):
        try:
            await cls.__db.close()
        except:
            pass


    @classmethod
    async def get_samples(cls, run, dataset, lumi=0):
        """
        If lumi is None or 0, only per run samples will be returned.
        If lumi is -1, all per lumi samples will be returned and per run samples will not be returned.
        If lumi is greater than 0, samples matching that lumi will be returned.
        """

        if run != None:
            run = '%%%s%%' % run
        if dataset != None:
            dataset = '%%%s%%' % dataset

        sql = 'SELECT DISTINCT run, dataset, lumi FROM samples '
        args = ()

        def add_where_condition(condition, arg):
            nonlocal sql
            nonlocal args
            sql += 'WHERE ' if len(args) == 0 else 'AND '
            sql += condition + ' '
            args += (arg,)

        if dataset != None:
            add_where_condition('dataset LIKE ?', dataset)
        if run != None:
            add_where_condition('run LIKE ?', run)

        if lumi == None or int(lumi) == 0:
            add_where_condition('lumi = ?', 0)
        elif int(lumi) == -1:
            add_where_condition('lumi != ?', 0)
        elif int(lumi) > 0:
            lumi_pattern = '%%%s%%' % lumi
            add_where_condition('lumi LIKE ?', lumi_pattern)

        cursor = await cls.__db.execute(sql, args)
        rows = await cursor.fetchall()
        await cursor.close()

        return rows


    @classmethod
    async def get_me_names_list(cls, dataset, run, lumi):
        """
        me_list format for an ME (ME/Path/mename) that has 2 QTests and an efficiency flag set looks like this:
        ME/Path/mename
        ME/Path/mename\0.qtest1
        ME/Path/mename\0.qtest2
        ME/Path/mename\0e=1
        This function returns a list of contents in provided run/dataset combination, in a format explained above.
        """

        # For now adding a LIMIT 1 because there might be multiple version of the same file.
        sql = 'SELECT menameblob FROM samples JOIN menames ON samples.menamesid = menames.menamesid WHERE run = ? AND lumi = ? AND dataset = ? LIMIT 1;'

        cursor = await cls.__db.execute(sql, (int(run), int(lumi), dataset))
        row = await cursor.fetchone()
        await cursor.close()

        if not row:
            # Blob doesn't exist, we should probably try to import
            return None

        names_list = await cls.compressor.uncompress_names_blob(row[0])
        return names_list


    @classmethod
    async def get_me_infos_list(cls, dataset, run, lumi):
        """Returns a list of MEInfo objects"""

        # For now adding a LIMIT 1 because there might be multiple version of the same file.
        sql = 'SELECT meinfosblob FROM samples JOIN meinfos ON samples.meinfosid = meinfos.meinfosid WHERE run = ? AND lumi = ? AND dataset = ? LIMIT 1;'

        cursor = await cls.__db.execute(sql, (int(run), int(lumi), dataset))
        row = await cursor.fetchone()
        await cursor.close()

        if not row:
            # Blob doesn't exist, we should probably try to import
            return None

        infos_list = await cls.compressor.uncompress_infos_blob(row[0])
        return infos_list


    @classmethod
    async def get_filename_fileformat_names_infos(cls, dataset, run, lumi=0):
        # For now adding a LIMIT 1 because there might be multiple version of the same file.
        sql = '''
        SELECT
            filename,
            fileformat,
            menameblob,
            meinfosblob
        FROM
            samples
            JOIN menames ON samples.menamesid = menames.menamesid
            JOIN meinfos ON samples.meinfosid = meinfos.meinfosid
        WHERE
            run = ?
        AND 
            lumi = ?
        AND 
            dataset = ?
        LIMIT 1;
        '''

        cursor = await cls.__db.execute(sql, (int(run), int(lumi), dataset))
        row = await cursor.fetchone()
        await cursor.close()

        if not row:
            # Blobs don't exist, we should probably try to import
            return None

        filename = row[0]
        fileformat = row[1]
        names_list = await cls.compressor.uncompress_names_blob(row[2])
        infos_list = await cls.compressor.uncompress_infos_blob(row[3])

        return (filename, fileformat, names_list, infos_list)

    
    @classmethod
    async def is_samples_empty(cls):
        """Returns True if at least one sample exists, False otherwise."""

        sql = 'SELECT COUNT(*) FROM (SELECT * FROM samples LIMIT 1);'
        cursor = await cls.__db.execute(sql)
        row = await cursor.fetchone()
        await cursor.close()
        return row[0] == 0


    @classmethod
    async def get_lumis(cls, dataset, run):
        """Returns all registered lumis for dataset/run combination"""
        
        sql = 'SELECT lumi FROM samples WHERE dataset = ? AND run = ? ORDER BY lumi;'
        cursor = await cls.__db.execute(sql, (dataset, int(run)))
        rows = await cursor.fetchall()
        await cursor.close()
        return [x[0] for x in rows]


    @classmethod
    async def get_sample_file_info(cls, dataset, run, lumi=0):
        """
        Returns a full path and a format of a ROOT file containing plots of a given sample. 
        Returns None if given sample doesn't exist.
        """

        sql = 'SELECT filename, fileformat FROM samples WHERE dataset = ? AND run = ? AND lumi = ?;'
        cursor = await cls.__db.execute(sql, (dataset, int(run), int(lumi)))
        row = await cursor.fetchone()

        await cursor.close()

        if row:
            return row[0], row[1]
        else:
            return None, None


    @classmethod
    async def register_samples(cls, samples):
        """
        Adds multiple samples to the database.
        Sample is declared here: data_types.SampleFull
        """

        sql = 'INSERT OR REPLACE INTO samples(dataset, run, lumi, filename, fileformat, menamesid, meinfosid) VALUES(?,?,?,?,?,NULL,NULL);'
        try:
            await cls.__lock.acquire()
            await cls.__db.execute('BEGIN;')

            await cls.__db.executemany(sql, samples)
            await cls.__db.execute('COMMIT;')
        except Exception as e:
            print(e)
            try:
                await cls.__db.execute('ROLLBACK;')
            except:
                pass
        finally:
            cls.__lock.release()


    @classmethod
    async def search_dataset_names(cls, search):
        """Returns at most 100 dataset names matching the search term."""

        search = '%%%s%%' % search.lower()
        sql = 'SELECT DISTINCT dataset FROM samples WHERE dataset LIKE ? LIMIT 100;'
        cursor = await cls.__db.execute(sql, (search,))
        rows = await cursor.fetchall()
        await cursor.close()
        return [x[0] for x in rows]


    @classmethod
    async def search_runs(cls, search):
        """Returns at most 100 run numbers matching the search term."""

        search = '%%%s%%' % search
        sql = 'SELECT DISTINCT run FROM samples LIMIT 100;'
        cursor = await cls.__db.execute(sql)
        rows = await cursor.fetchall()
        await cursor.close()
        return [x[0] for x in rows]


    @classmethod
    async def get_latest_runs(cls, search):
        """Returns at most 100 latest run numbers."""

        sql = 'SELECT DISTINCT run FROM samples ORDER BY run DESC LIMIT 100;'
        args = ()
        if search != None and search != '':
            search = '%%%s%%' % search
            sql = 'SELECT DISTINCT run FROM samples WHERE run LIKE ? ORDER BY run DESC LIMIT 100;'
            args = (search,)

        cursor = await cls.__db.execute(sql, args)
        rows = await cursor.fetchall()
        await cursor.close()
        return [x[0] for x in rows]


    @classmethod
    async def add_blobs(cls, names_blob, infos_blob, dataset, filename, run, lumi=0):
        """Adds me list blob and infos blob to a DB in a single transaction."""

        try:
            await cls.__lock.acquire()
            await cls.__db.execute('BEGIN;')

            await cls.__db.execute('INSERT OR IGNORE INTO menames (menameblob) VALUES (?);', (names_blob,))
            cursor = await cls.__db.execute('SELECT menamesid FROM menames WHERE menameblob = ?;', (names_blob,))
            row = await cursor.fetchone()
            names_blob_id = row[0]

            await cls.__db.execute('INSERT OR IGNORE INTO meinfos (meinfosblob) VALUES (?);', (infos_blob,))
            cursor = await cls.__db.execute('SELECT meinfosid FROM meinfos WHERE meinfosblob = ?;', (infos_blob,))
            row = await cursor.fetchone()
            infos_blob_id = row[0]

            sql = 'UPDATE samples SET menamesid = ?, meinfosid = ? WHERE run = ? AND lumi = ? AND dataset = ? AND filename = ?;'
            await cls.__db.execute(sql, (names_blob_id, infos_blob_id, int(run), int(lumi), dataset, filename))

            await cls.__db.execute('COMMIT;')
        except Exception as e:
            print(e)
            try:
                await cls.__db.execute('ROLLBACK;')
            except:
                pass
        finally:
            cls.__lock.release()
