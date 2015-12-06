#!/usr/bin/env python
"""

Example script to test writing to local sqlite db.

"""

import sys
from CondCore.Utilities.CondDBFW import querying_framework_api
from CondCore.Utilities.CondDBFW import querying

class query_script():
	def script(self, connection):
		payload = connection.payload(hash=sys.argv[1])
		return payload

connection_data = {"db_alias" : "orapro", "schema" : "cms_conditions", "host" : "oracle", "secrets" : "/afs/cern.ch/cms/DB/conddb/.cms_cond/netrc"}
qf = querying_framework_api(connection_data)
data = qf.run_script(query_script())

import pprint
pprint.pprint(data.as_dicts())

# test writing to sqlite database

sqlite_db_url = "/tmp/jdawes/CMSSW_7_5_2/src/CondCore/Utilities/python/CondDBFW/examples/sqlite_tests.sqlite"
sqlite_con = querying.connect({"host" : "sqlite", "db_alias" : sqlite_db_url})

sqlite_con.write(data)
sqlite_con.commit()