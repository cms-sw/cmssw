#!/usr/bin/env python

'''
'''

__author__ = 'Giacomo Govi'

import CondCore.Utilities.o2o as o2olib
import sys
import optparse
import argparse
from sets import Set

class CommandTool(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.options = Set()
        self.commands = {}
        self.args = None
    
    def addOption( self, *keys, **params ):
        nkeys = []
        if len(keys)>1:
            nkeys.append( "-"+keys[1] )
        nkeys.append( "--"+keys[0] )
        action = "store"
        if 'type' not in params.keys():
            params['action'] = "store_true"
        self.parser.add_argument(*nkeys,**params ) 
        self.options.add( keys[0] )

    def addCommand( self, command_name, help_entry, *requiredOptions ):
        self.parser.add_argument("--"+command_name, action="store_true", help=help_entry )
        for opt in requiredOptions:
            if opt not in self.options:
                raise Exception("Option '%s' has not been registered." %opt )
        self.commands[command_name] = requiredOptions

    def setup():
        return

    def execute(self):
        self.args = self.parser.parse_args()
        executed = False
        self.setup()
        for k in self.commands.keys():
            if getattr(self.args,k):
                if executed:
                    print 'Ignoring command %s...' %k
                else:
                    required_options = self.commands[k]
                    for o in required_options:
                        val = getattr(self.args,o)
                        if val is None:
                            raise Exception( "Required option '%s' has not been specified." %o )
                    func = getattr(self,k)
                    func()
                    executed = True
        return executed

class O2OMgrTool(CommandTool):
    def __init__(self):
        CommandTool.__init__(self)
        self.mgr = None
        CommandTool.addOption(self,"name", "n", type=str, help="the o2o job name")
        CommandTool.addOption(self,"configFile", "c", type=str, help="the JSON configuration file path")
        CommandTool.addOption(self,"interval", "i", type=int, help="the chron job interval")
        CommandTool.addOption(self,"db", type=str, help="the target database: pro ( for prod ) or dev ( for prep ). default=pro")
        CommandTool.addOption(self,"auth","a", type=str,  help="path of the authentication file")
        CommandTool.addCommand( self,"create", "create a new O2O job", "name","configFile","interval")
        CommandTool.addCommand(self,"setConfig","set a new configuration for the specified job","name","configFile" )
        CommandTool.addCommand(self,"setInterval","set a new execution interval for the specified job","name","interval" )
        CommandTool.addCommand(self,"enable","enable the O2O job","name" )
        CommandTool.addCommand(self,"disable", "disable the O2O job" , "name")
        CommandTool.addCommand(self,"migrate", "migrate the tag info for the jobs in configuration entries" )
        CommandTool.addCommand(self,"listJobs", "list the registered jobs" )
        CommandTool.addCommand(self,"listConf", "shows the configurations for the specified job", "name")

    def setup(self):
        db_service = o2olib.prod_db_service
        if self.args.db is not None:
            if self.args.db == 'dev' or self.args.db == 'oradev' :
                db_service = o2olib.dev_db_service
            elif self.args.db != 'orapro' and self.args.db != 'onlineorapro' and self.args.db != 'pro':
                raise Exception("Database '%s' is not known." %self.args.db )
        
        self.mgr = o2olib.O2OJobMgr()
        return self.mgr.connect( db_service, self.args.auth )
        
    def create(self):
        self.mgr.add( self.args.name, self.args.configFile, self.args.interval, True )

    def setConfig(self):
        self.mgr.setConfig( self.args.name, self.args.configFile )

    def setInterval(self):
        self.mgr.setConfig( self.args.name, self.args.interval )

    def enable(self):
        self.mgr.setConfig( self.args.name, True )
    
    def disable(self):
        self.mgr.setConfig( self.args.name, False )

    def migrate(self):
        self.mgr.migrateConfig()

    def listJobs(self):
        self.mgr.listJobs()

    def listConf(self):
        self.mgr.listConfig( self.args.name )

def main( argv ):

    tool = O2OMgrTool()
    ret = False
    try:
        ret = tool.execute()
    except Exception as e:
        print e
    return ret

if __name__ == '__main__':

    sys.exit(main(sys.argv))
