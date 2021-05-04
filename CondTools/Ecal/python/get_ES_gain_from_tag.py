# Script to extract the gain used by ES providing the TAG of the run key.
# I needs in this order:
# - the tag name

__author__ = 'Giacomo Cucciati'

import sys
import cx_Oracle

import CondTools.Ecal.db_credentials as auth

db_target_account = 'CMS_ES_CONF'
db_service,db_user,db_pwd = auth.get_db_credentials( db_target_account )
conn_str = u'%s/%s@%s' %(db_user,db_pwd,db_service)
def main(argv):
 
  if len(argv) !=2:
    print("Wrong number of parameters")
    return
  
  runKey_tag = argv[1]
  conn = cx_Oracle.connect(conn_str)
  conn.current_schema = db_target_account
  c = conn.cursor()

  sql_query = "select max(version) from ES_RUN_CONFIGURATION_DAT where tag='"
  sql_query += runKey_tag + "'"
  c.execute(sql_query)
  print_row = []
  for row in c:
    print_row.append(str(row[0]))
  runKey_max_version = ', '.join(print_row)
 
  sql_query  = "select ES_SEQUENCE_DAT.SEQUENCE_ID from ES_SEQUENCE_DAT "
  sql_query += "inner join ES_RUN_CONFIGURATION_DAT on ES_SEQUENCE_DAT.ES_CONFIG_ID=ES_RUN_CONFIGURATION_DAT.CONFIG_ID "
  sql_query += "and ES_RUN_CONFIGURATION_DAT.tag='" + runKey_tag + "' and version=" + runKey_max_version;
  c.execute(sql_query)
  print_row = []
  for row in c:
    print_row.append(str(row[0]))
  sequence_id = ', '.join(print_row)

  sql_query  = "select ES_CCS_CONFIGURATION.GAIN from ES_CCS_CONFIGURATION "
  sql_query += "inner join ES_CYCLE_DAT on ES_CCS_CONFIGURATION.CCS_CONFIGURATION_ID=ES_CYCLE_DAT.CCS_CONFIGURATION_ID "
  sql_query += "and ES_CYCLE_DAT.SEQUENCE_ID=" + sequence_id;
  c.execute(sql_query)
  print_row = []
  for row in c:
    print_row.append(str(row[0]))
  gain = ', '.join(print_row)
  print(gain)

  conn.close()

if __name__ == "__main__":
  main(sys.argv)

