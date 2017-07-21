import re
import os, sys, shutil
import subprocess
"""
A simple helper script that provided with no arguments dumps a list of
top-level keys, and provided with any key from this list as an argument,
dumps a list of sub-keys and saves corresponding configuration to local
files.
"""

# connection string
sqlplusCmd = ['env',
              'sqlplus',
              '-S',
              'cms_trg_r/@cms_omds_adg'
             ]

if hash( sqlplusCmd[-1] ) != 1687624727082866629:
    print 'Do not forget to plug password to this script'
    print 'Exiting.'
    exit(0)

myre = re.compile(r'(ID)|(-{80})')

# if no arguments are given, query the top level keys only and exit
if len(sys.argv) == 1:
    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    print 'No args specified, querying and printing only top-level config keys:'
    for line in re.split('\n',sqlplus.communicate('select unique ID from CMS_TRG_L1_CONF.UGT_KEYS;')[0]):
        if myre.search(line) == None :
            print line
    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    print 'No args specified, querying and printing only top-level Run Settings keys:'
    for line in re.split('\n',sqlplus.communicate('select unique ID from CMS_TRG_L1_CONF.UGT_RS_KEYS;')[0]):
        if myre.search(line) == None :
            print line
    print 'Pick any of these keys as an argument next time you run this script'
    exit(0)

# if an argument is given query the whole content of the key
key = sys.argv[1]
rsKey = sys.argv[2]

# if the argument is the very top level key, querry the uGT specific subkey
queryKey   = "select UGT_KEY    from CMS_TRG_L1_CONF.L1_TRG_CONF_KEYS where ID='{0}'".format(key)
queryRsKey = "select UGT_RS_KEY from CMS_TRG_L1_CONF.L1_TRG_RS_KEYS   where ID='{0}'".format(rsKey)

sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
for line in re.split('\n',sqlplus.communicate(queryKey+';')[0]):
    print line
    if re.search('/v',line) :
        key=line

sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
for line in re.split('\n',sqlplus.communicate(queryRsKey+';')[0]):
    print line
    if re.search('/v',line) :
        rsKey=line

print key+":"+rsKey

queryKeys = """
            select
                ALGOBX_MASK, ALGO_FINOR_MASK, ALGO_FINOR_VETO, ALGO_PRESCALE
            from
                CMS_TRG_L1_CONF.UGT_RS_KEYS
            where
                ID = '{0}'
            """.format(rsKey)

with open("menu.xml",'w') as f:
    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    query = """
            select
                CONF
            from
                CMS_TRG_L1_CONF.UGT_L1_MENU MENU
            inner join
            (
            select
                L1_MENU
            from
                CMS_TRG_L1_CONF.UGT_KEYS
            where
                ID = '{0}'
            ) TOP_KEYS
            on 
                TOP_KEYS.L1_MENU = MENU.ID
            """.format(key)
    for line in re.split('\n',sqlplus.communicate('\n'.join(['set linesize 20000', 'set longchunksize 20000000 long 20000000 pages 0',query+';']))[0]):
        f.write('\n')
        f.write(line)
    f.close()

# write results for specific configs to the following files
batch = {
         'ALGOBX_MASK'     : 'algobx_mask.xml',
         'ALGO_FINOR_MASK' : 'algo_finor_mask.xml',
         'ALGO_FINOR_VETO' : 'algo_finor_veto.xml',
         'ALGO_PRESCALE'   : 'algo_prescale.xml'
        }

# do the main job here
for config,fileName in batch.iteritems():

    sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    with open(fileName,'w') as f:
        query = """
                select
                    CLOBS.CONF
                from
                    CMS_TRG_L1_CONF.UGT_RS_CLOBS CLOBS, ({0}) KEY
                where
                    CLOBS.ID = KEY.{1}
                """.format(queryKeys, config)

        for line in re.split('\n',sqlplus.communicate('\n'.join(['set linesize 200', 'set longchunksize 200000 long 200000 pages 0',query+';']))[0]):
            f.write('\n')
            f.write(line)
        f.close()


sqlplus = subprocess.Popen(sqlplusCmd, shell=False, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
print 'Following keys were found:'
for line in re.split('\n',sqlplus.communicate(queryKeys+';')[0]):
    print line

print 'Results are saved in ' + ' '.join(batch.values()) + ' files'

