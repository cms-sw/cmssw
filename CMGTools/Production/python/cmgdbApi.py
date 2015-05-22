#!/usr/bin/env python
## Author: Peter Meckiffe
## @ CERN, Meyrin
## September 27th 2011

# This module will write the data from a passed software object
# to the database file on disk
import os
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/oracle/11.2.0.1.0p2/etc/profile.d/init.sh")
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/python/2.6.4-cms16/etc/profile.d/init.sh")
os.system("source /afs/cern.ch/cms/slc5_amd64_gcc434/external/py2-cx-oracle/5.1/etc/profile.d/init.sh")

import sys, re
#import CMGTools.Production.cx_Oracle as cx_Oracle
import cx_Oracle

class CmgdbApi(object):

    """ A class for interacting with the CMGDB database """
    
    def __init__(self,development=False):

        self.insertConn = None
        self.selectConn = None
        self.selectCur = None
        self.insertCur = None
        self._masterCur = None
        self.development = development
        if self.development==False:
            self.schema_name="cms_cmgdb"
        else:
            self.schema_name="cmgbookkeepingtest"
        
    def connect(self):
        """
        Create an object and connect to CMGDB

        No arguments are required to initialise the object
        Two database connections are opened,
        insertConn = a connection object for insert queries,
        selectConn = a connection object for insert queries
        """
        try:
            if self.development == False:
                # Connect to insert account
                self.insertConn = cx_Oracle.connect("cms_cmgdb_w/NGU79myr231@(DESCRIPTION=(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr1-s.cern.ch) (PORT=10121) )(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr2-s.cern.ch) (PORT=10121) )(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr3-s.cern.ch) (PORT=10121) )(LOAD_BALANCE=on)(ENABLE=BROKEN)(CONNECT_DATA=(SERVER=DEDICATED)(SERVICE_NAME=cmsr_lb.cern.ch) (FAILOVER_MODE = (TYPE = SELECT)(METHOD = BASIC)(RETRIES = 200)(DELAY = 15))))")
                # Connect to select account
                self.selectConn = cx_Oracle.connect("cms_cmgdb_r/NGU79myr231@(DESCRIPTION=(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr1-s.cern.ch) (PORT=10121) )(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr2-s.cern.ch) (PORT=10121) )(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr3-s.cern.ch) (PORT=10121) )(LOAD_BALANCE=on)(ENABLE=BROKEN)(CONNECT_DATA=(SERVER=DEDICATED)(SERVICE_NAME=cmsr_lb.cern.ch) (FAILOVER_MODE = (TYPE = SELECT)(METHOD = BASIC)(RETRIES = 200)(DELAY = 15))))")
                self._masterConn = cx_Oracle.connect("cms_cmgdb/NGU79myr231@(DESCRIPTION=(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr1-s.cern.ch) (PORT=10121) )(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr2-s.cern.ch) (PORT=10121) )(ADDRESS= (PROTOCOL=TCP) (HOST=cmsr3-s.cern.ch) (PORT=10121) )(LOAD_BALANCE=on)(ENABLE=BROKEN)(CONNECT_DATA=(SERVER=DEDICATED)(SERVICE_NAME=cmsr_lb.cern.ch) (FAILOVER_MODE = (TYPE = SELECT)(METHOD = BASIC)(RETRIES = 200)(DELAY = 15))))")
                # Create cx_Oracle cursor objects from connections
                self.selectCur = self.selectConn.cursor()
                self._masterCur = self._masterConn.cursor()
                self.insertCur = self.insertConn.cursor()
                self.selectCur.execute("ALTER SESSION SET CURRENT_SCHEMA=CMS_CMGDB")
                self.insertCur.execute("ALTER SESSION SET CURRENT_SCHEMA=CMS_CMGDB")
            else:
                # Connect to insert account
                self.insertConn = cx_Oracle.connect("cmgbookkeepingtest/NGU79myr231@(DESCRIPTION=(ADDRESS= (PROTOCOL=TCP) (HOST=dbsrvg3305.cern.ch) (PORT=10121) )(LOAD_BALANCE=on)(ENABLE=BROKEN)(CONNECT_DATA=(SERVER=DEDICATED)(SID=DEVDB11) (FAILOVER_MODE = (TYPE = SELECT)(METHOD = BASIC)(RETRIES = 200)(DELAY = 15))))")
                # Connect to select account
                self.selectConn = cx_Oracle.connect("cmgbookkeepingtest/NGU79myr231@(DESCRIPTION=(ADDRESS= (PROTOCOL=TCP) (HOST=dbsrvg3305.cern.ch) (PORT=10121) )(LOAD_BALANCE=on)(ENABLE=BROKEN)(CONNECT_DATA=(SERVER=DEDICATED)(SID=DEVDB11) (FAILOVER_MODE = (TYPE = SELECT)(METHOD = BASIC)(RETRIES = 200)(DELAY = 15))))")
                self._masterConn = cx_Oracle.connect("cmgbookkeepingtest/NGU79myr231@(DESCRIPTION=(ADDRESS= (PROTOCOL=TCP) (HOST=dbsrvg3305.cern.ch) (PORT=10121) )(LOAD_BALANCE=on)(ENABLE=BROKEN)(CONNECT_DATA=(SERVER=DEDICATED)(SID=DEVDB11) (FAILOVER_MODE = (TYPE = SELECT)(METHOD = BASIC)(RETRIES = 200)(DELAY = 15))))")
                # Create cx_Oracle cursor objects from connections
                self.selectCur = self.selectConn.cursor()
                self._masterCur = self._masterConn.cursor()
                self.insertCur = self.insertConn.cursor()
                self.selectCur.execute("ALTER SESSION SET CURRENT_SCHEMA=CMGBOOKKEEPINGTEST")
                self.insertCur.execute("ALTER SESSION SET CURRENT_SCHEMA=CMGBOOKKEEPINGTEST")
        except Exception as dbError:
            print "Unable to connect to CMGDB"
            print dbError.args[0]
            exit( -1 )

    # Return DB description as a string
    def describe(self):
        tables = []
        description = ""
        """Returns a description of the database as a string"""
        self._masterCur.execute("SELECT table_name from tabs")
        for table in self._masterCur:
            tables.append(table)
        for table in tables:
            description += table[0] + "\n"
            self._masterCur.execute("select * from "+table[0]+" where 1 = 0")

            for column in self._masterCur.description:
                description += "\t-"+column[0]+"\n"
        return description

    # Pass an SQL select query
    def sql(self, query):
        """Pass an SQL query to CMGDB
        'query' takes the SQL query as a string
        """
        columns = []
        rows = []
        self.selectCur.execute(query)
        for column in self.selectCur.description:
            columns.append(column[0])
        for row in self.selectCur:
            rows.append(row)
        return columns, rows


    # Close connection with database and destroy object
    def close(self):
        """Close database connections"""
        self.insertConn.close()
        self.selectConn.close()
