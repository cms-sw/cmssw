#!/usr/bin/env python

import pprint
import datetime
import sys

if __name__ == "__main__":
	# this class should extend a framework base class that
	# provides default implementations of important methods
	import sys
	sys.path.append("../")
	from CondCore.Utilities.CondDBFW import querying_framework_api
	import CondCore.Utilities.CondDBFW.data_sources, CondCore.Utilities.CondDBFW.data_formats as format
	from CondCore.Utilities.CondDBFW.querying import connect

	class query_script():

		def script(self, connection):
			tag = connection.tag(name=sys.argv[1])
			iovs = tag.iovs(pretty=True)
			return iovs

	secrets_file = "/afs/cern.ch/cms/DB/conddb/.cms_cond/netrc"
	secrets_file_1 = "netrc_test"

	connection_data = {"db_alias" : "orapro", "schema" : "cms_conditions", "host" : "oracle", "secrets" : secrets_file}
	qf = querying_framework_api(connection_data)
	data = qf.run_script(query_script())

	data.as_table(columns=["since", "payload_hash", "insertion_time"], fit=["all"])