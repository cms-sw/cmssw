# taken  from 
"""
http://wiki.python.org/moin/ConfigParserExamples
http://stackoverflow.com/questions/3220670/read-all-the-contents-in-ini-file-into-dictionary-with-python
"""
from __future__ import print_function

import configparser as cp 

Config=cp.ConfigParser()

def ConfigSectionMap(section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1
