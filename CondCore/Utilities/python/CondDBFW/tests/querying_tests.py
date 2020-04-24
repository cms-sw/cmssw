import unittest
import sys
import datetime
import pprint
import subprocess
import os

import CondCore.Utilities.CondDBFW.querying as querying
import CondCore.Utilities.CondDBFW.data_sources as data_sources
import CondCore.Utilities.CondDBFW.data_formats as data_formats
import CondCore.Utilities.CondDBFW.shell as shell
import CondCore.Utilities.CondDBFW.models as models

from CondCore.Utilities.CondDBFW.utils import to_timestamp, to_datetime, friendly_since
from CondCore.Utilities.CondDBFW.models import Range, Radius

prod_connection_string = "frontier://FrontierProd/CMS_CONDITIONS"
secrets_source = None

class querying_tests(unittest.TestCase):
	def setUp(self):
		self.connection = querying.connect(prod_connection_string, secrets=secrets_source)
		self.global_tag_name = "74X_dataRun1_HLT_frozen_v2"
	def test_check_connection(self):
		self.assertTrue(self.connection != None)
	def tearDown(self):
		self.connection.tear_down()

def factory_tests(querying_tests):

	"""
	Factory
	"""

	def test_check_keys_in_models(self):
		"""
		Verifies that each key that is required is present in self.connection's model dictionary,
		and also that self.connection has a proxy method for each model that is generated.
		"""
		keys = ["globaltag", "globaltagmap", "tag", "iov", "payload"]
		for key in keys:
			self.assertTrue(key in self.connection.models.keys())
		proxy_method_names = ["global_tag", "global_tag_map", "tag", "iov", "payload"]
		for name in proxy_method_names:
			self.assertTrue(hasattr(self.connection, name))

	def test_raw_factory_calls_all_models(self):
		"""
		Verifies that the factory object held by the connection generates classes belonging to the type
		held by self.connection.
		"""
		keys = ["globaltag", "globaltagmap", "tag", "iov", "payload"]
		for key in keys:
			self.assertTrue(isinstance(self.connection.factory.object(key), self.connection.models[key]))

def global_tag_tests(querying_tests):

	"""
	Global Tags
	"""

	def test_get_global_tag(self):
		# hard coded global tag for now
		global_tag = self.connection.global_tag(name=self.global_tag_name)
		self.assertTrue(global_tag.name != None)
		self.assertTrue(global_tag.tags() != None)
		self.assertTrue(isinstance(global_tag, self.connection.models["globaltag"]))

	def test_get_empty_global_tag(self):
		empty_gt = self.connection.global_tag()
		self.assertTrue(isinstance(empty_gt, self.connection.models["globaltag"]))
		self.assertTrue(empty_gt.empty)

	def test_all_global_tags_empty_gt(self):
		empty_gt = self.connection.global_tag()
		all_gts = empty_gt.all(amount=10)
		self.assertTrue(all_gts != None)
		self.assertTrue(len(all_gts.data()) != 0)

	def test_all_method_parameters(self):
		if self.connection.connection_data["host"].lower() == "frontier":
			print("Cannot query for timestamps on frontier connection.")
			return
		empty_gt = self.connection.global_tag()
		sample_time = datetime.datetime(year=2016, month=1, day=1)
		now = datetime.datetime.now()
		time_range = Range(sample_time, now)
		time_radius = Radius(sample_time, datetime.timedelta(weeks=4))
		all_gts_in_interval = empty_gt.all(insertion_time=time_range).data()
		for gt in all_gts_in_interval:
			self.assertTrue(sample_time <= gt.insertion_time <= now)

	def test_as_dicts_method_without_timestamps_conversion(self):
		global_tag = self.connection.global_tag(name=self.global_tag_name)
		dict_form = global_tag.as_dicts(convert_timestamps=False)
		self.assertTrue(isinstance(dict_form, dict))
		self.assertTrue(dict_form["insertion_time"], datetime.datetime)
		self.assertTrue(dict_form["snapshot_time"], datetime.datetime)

	def test_as_dicts_method_with_timestamps_conversion(self):
		global_tag = self.connection.global_tag(name=self.global_tag_name)
		dict_form = global_tag.as_dicts(convert_timestamps=True)
		self.assertTrue(isinstance(dict_form, dict))
		self.assertTrue(dict_form["insertion_time"], str)
		self.assertTrue(dict_form["snapshot_time"], str)

		# now check that the timestamp encoded in the strings is the same as the
		# datetime object that were converted
		self.assertTrue(to_datetime(dict_form["insertion_time"]) == global_tag.insertion_time)
		self.assertTrue(to_datetime(dict_form["snapshot_time"]) == global_tag.snapshot_time)

	def test_get_tag_maps_in_global_tag(self):
		global_tag = self.connection.global_tag(name=self.global_tag_name)

		# get global tag maps
		global_tag_maps = global_tag.tags()
		self.assertTrue(isinstance(global_tag_maps, data_sources.json_list))
		global_tag_maps_list = global_tag_maps.data()
		self.assertTrue(isinstance(global_tag_maps_list, list))
		self.assertTrue(len(global_tag_maps_list) != 0)

		for gt_map in global_tag_maps_list:
			self.assertTrue(isinstance(gt_map, self.connection.models["globaltagmap"]))

	def test_get_tag_maps_in_global_tag_with_parameters(self):
		global_tag = self.connection.global_tag(name=self.global_tag_name)

		global_tag_maps_specific_record = global_tag.tags(record="AlCaRecoTriggerBitsRcd")

		self.assertTrue(isinstance(global_tag_maps_specific_record, data_sources.json_list))

		gt_maps_spec_record_list = global_tag_maps_specific_record.data()
		self.assertTrue(isinstance(gt_maps_spec_record_list, list))
		# this global tag is old, so this test is unlikely to fail since
		# its global tag maps will not fail
		self.assertTrue(len(gt_maps_spec_record_list) == 3)

	def test_get_all_iovs_within_range(self):
		global_tag = self.connection.global_tag(name=self.global_tag_name)

		since_range = self.connection.range(200000, 300000)
		iovs = global_tag.iovs(since=since_range)
		self.assertTrue(isinstance(iovs, data_sources.json_list))
		iovs_list = iovs.data()
		self.assertTrue(isinstance(iovs_list, list))

		for iov in iovs_list:
			self.assertTrue(isinstance(iov, self.connection.models["iov"]))
			self.assertTrue(200000 <= iov.since <= 300000)

	def test_gt_diff(self):
		gt1 = self.connection.global_tag(name="74X_dataRun1_v1")
		gt2 = self.connection.global_tag(name="74X_dataRun1_v3")
		difference_by_arithmetic = gt1 - gt2
		difference_by_method = gt1.diff(gt2)

		# verify both methods of difference
		self.assertTrue(isinstance(difference_by_arithmetic, data_sources.json_list))
		self.assertTrue(isinstance(difference_by_method, data_sources.json_list))

		difference_arithmetic_list = difference_by_arithmetic.data()
		difference_method_list = difference_by_method.data()
		self.assertTrue(isinstance(difference_arithmetic_list, list))
		self.assertTrue(isinstance(difference_method_list, list))

		self.assertTrue(len(difference_arithmetic_list) == len(difference_method_list))

		for n in range(len(difference_arithmetic_list)):
			self.assertTrue(difference_arithmetic_list[n]["%s Tag" % gt1.name] != difference_arithmetic_list[n]["%s Tag" % gt2.name])
			self.assertTrue(difference_method_list[n]["%s Tag" % gt1.name] != difference_method_list[n]["%s Tag" % gt2.name])

class global_tag_map_tests(querying_tests):


	"""
	Global Tag Map
	"""

	def test_get_global_tag_map(self):
		global_tag_name = "74X_dataRun1_HLT_frozen_v2"
		tag_name = "AlCaRecoTriggerBits_MuonDQM_v1_hlt"
		gt_map = self.connection.global_tag_map(global_tag_name=self.global_tag_name, tag_name=tag_name)
		self.assertTrue(isinstance(gt_map, self.connection.models["globaltagmap"]))

	def test_get_empty_global_tag_map(self):
		empty_gt_map = self.connection.global_tag_map()
		self.assertTrue(isinstance(empty_gt_map, self.connection.models["globaltagmap"]))
		self.assertTrue(empty_gt_map.empty)

class tag_tests(querying_tests):

	"""
	Tags
	"""

	def test_get_tag(self):
		# hard code tag for now
		tag_name = "EBAlignment_measured_v01_express"
		tag = self.connection.tag(name=tag_name)
		# this tag exists, so shouldn't be NoneType
		# we gave the name, so that should at least be in the tag object
		self.assertTrue(tag.name != None)
		self.assertTrue(isinstance(tag, self.connection.models["tag"]))

	def test_get_empty_tag(self):
		tag = self.connection.tag()
		self.assertTrue(isinstance(tag, self.connection.models["tag"]))
		self.assertTrue(tag.empty)

	def test_get_parent_global_tags(self):
		tag_name = "EBAlignment_measured_v01_express"
		tag = self.connection.tag(name=tag_name)
		self.assertTrue(tag != None)
		parent_global_tags = tag.parent_global_tags()
		self.assertTrue(parent_global_tags != None)

	def test_all_tags_empty_tag(self):
		empty_tag = self.connection.tag()
		all_tags = empty_tag.all(amount=10)
		self.assertTrue(all_tags != None)
		# there are always tags in the db
		self.assertTrue(len(all_tags.data()) != 0)

	def test_all_tags_non_empty_tag(self):
		tag_name = "EBAlignment_measured_v01_express"
		empty_tag = self.connection.tag(name=tag_name)
		all_tags = empty_tag.all(amount=10)
		self.assertTrue(all_tags != None)
		# there are always tags in the db
		self.assertTrue(len(all_tags.data()) != 0)

class iov_tests(querying_tests):

	"""
	IOVs
	"""

	def test_get_iov(self):
		tag_name = "EBAlignment_measured_v01_express"
		tag = self.connection.tag(name=tag_name)
		iovs = tag.iovs()
		self.assertTrue(isinstance(iovs, data_sources.json_list))
		raw_list = iovs.data()
		self.assertTrue(isinstance(raw_list, list))
		first_iov = raw_list[0]
		self.assertTrue(isinstance(first_iov, self.connection.models["iov"]))

	def test_get_iovs_by_iov_query(self):
		tag_name = "EBAlignment_measured_v01_express"
		iovs = self.connection.iov(tag_name=tag_name)
		self.assertTrue(isinstance(iovs, data_sources.json_list))
		raw_list = iovs.data()
		self.assertTrue(isinstance(raw_list, list))
		first_iov = raw_list[0]
		self.assertTrue(isinstance(first_iov, self.connection.models["iov"]))

	def test_get_empty_iov(self):
		empty_iov = self.connection.iov()
		self.assertTrue(isinstance(empty_iov, self.connection.models["iov"]))
		self.assertTrue(empty_iov.empty)

class payload_tests(querying_tests):

	"""
	Payloads
	"""

	def test_get_payload(self):
		# hard coded payload for now
		payload_hash = "00172cd62d8abae41915978d815ae62cc08ad8b9"
		payload = self.connection.payload(hash=payload_hash)
		self.assertTrue(isinstance(payload, self.connection.models["payload"]))

	def test_get_empty_payload(self):
		payload = self.connection.payload()
		self.assertTrue(isinstance(payload, self.connection.models["payload"]))
		self.assertTrue(payload.empty)

	def test_get_parent_tags_payload(self):
		payload_hash = "00172cd62d8abae41915978d815ae62cc08ad8b9"
		payload = self.connection.payload(hash=payload_hash)
		self.assertTrue(payload != None)
		parent_tags = payload.parent_tags()
		self.assertTrue(parent_tags != None)

	def test_type_parent_tags(self):
		payload_hash = "00172cd62d8abae41915978d815ae62cc08ad8b9"
		payload = self.connection.payload(hash=payload_hash)
		self.assertTrue(payload != None)
		parent_tags = payload.parent_tags()
		self.assertTrue(isinstance(parent_tags, data_sources.json_list))

class misc_tests(querying_tests):

	"""
	Misc
	"""

	def test_search_everything(self):
		string_for_non_empty_result = "ecal"
		data = self.connection.search_everything(string_for_non_empty_result)
		self.assertTrue(len(data.get("global_tags").data()) != 0)
		self.assertTrue(len(data.get("tags").data()) != 0)
		self.assertTrue(len(data.get("payloads").data()) != 0)

class result_type_tests(querying_tests):

	"""
	Results types
	"""

	def test_factory_multiple_results(self):
		tags = self.connection.tag(time_type="Run")
		self.assertTrue(len(tags.data()) > 1)

	def test_factory_empty_result(self):
		tag = self.connection.tag()
		self.assertTrue(tag.empty)

	def test_factory_no_result(self):
		tag = self.connection.tag(name="dfasdf")
		self.assertTrue(tag == None)

	def tearDown(self):
		self.connection.tear_down()
