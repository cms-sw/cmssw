#!/usr/bin/env python
"""

Example script to show returning data as json.

"""

import sys
from CondCore.Utilities.CondDBFW import querying_framework_api

class query_script():
	def script(self, connection):
		tags = connection.tag().all(amount=sys.argv[1])
		return tags

connection_data = {"db_alias" : "orapro", "schema" : "cms_conditions", "host" : "oracle", "secrets" : "/afs/cern.ch/cms/DB/conddb/.cms_cond/netrc"}
qf = querying_framework_api(connection_data)
data = qf.run_script(query_script()).as_dicts()

import pprint
pprint.pprint(data)