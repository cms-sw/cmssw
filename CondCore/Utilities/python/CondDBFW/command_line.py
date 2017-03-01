#!/usr/bin/env python
"""
Command line module that the "command line" script.

Works by taking the main keyword (first command given to the script),
passing that to the function that will deal with that action, along with the following arguments as parameters for that function.
"""

import querying
import argparse
import datetime

def list_object(arguments):

	# set up connection
	connection = querying.connect(arguments.db, secrets=arguments.secrets, mode=arguments.mode)

	options = ["tag", "gt", "gts_for_tag"]
	number_of_options_given = 0
	for option in options:
		if getattr(arguments, option):
			number_of_options_given += 1
	if number_of_options_given != 1:
		print("You must specify a single object to list.")
		exit()

	if arguments.tag:
		tag_name = arguments.tag
		tag = connection.tag(name=tag_name)
		if tag:
			iovs = tag.iovs(amount=arguments.limit)
			iovs.as_table()
		else:
			print("The Tag '%s' was not found in the database '%s'." % (tag_name, arguments.db))
			exit()

	elif arguments.gt:
		gt_name = arguments.gt
		gt = connection.global_tag(name=gt_name)
		if gt:
			gt_maps = gt.tags(amount=arguments.limit)
			gt_maps.as_table(hide=["global_tag_name"])
		else:
			print("The Global Tag '%s' was not found in the database '%s'." % (gt_name, arguments.db))
			exit()

	elif arguments.gts_for_tag:
		tag_name = arguments.gts_for_tag
		tag = connection.tag(name=tag_name)
		gts = tag.parent_global_tags(amount=arguments.limit)
		gts.as_table(columns=["name", "insertion_time", "snapshot_time"])

def diff_of_tags(arguments):
	# get a CondDBFW Tag object for the first tag
	# then use the diff() method to draw the table of differences

	# set up connection
	connection = querying.connect(arguments.db, secrets=arguments.secrets, mode=arguments.mode)

	tag1 = connection.tag(name=arguments.tag1)
	tag2 = connection.tag(name=arguments.tag2)

	tag1.diff(tag2).as_table(columns=["since", arguments.tag1, arguments.tag2])

def diff_of_gts(arguments):
	# get a CondDBFW Global Tag object for the first GT
	# then use the diff() method to draw the table of differences

	# set up connection
	connection = querying.connect(arguments.db, secrets=arguments.secrets, mode=arguments.mode)

	gt1 = connection.global_tag(name=arguments.gt1)
	gt2 = connection.global_tag(name=arguments.gt2)

	gt1.diff(gt2).as_table(columns=["Record", "Label", "%s Tag" % arguments.gt1, "%s Tag" % arguments.gt2])

def search(arguments):

	raise NotImplementedError("Todo")

	connection = querying.connect(arguments.db, secrets=arguments.secrets, mode=arguments.mode)

	search_string = connection.regexp(".*%s.*" % arguments.string)

def copy_tag(arguments):

	# set up connection
	source_connection = querying.connect(arguments.db, secrets=arguments.secrets, mode=arguments.mode, map_blobs=True)
	dest_connection = querying.connect(arguments.dest_db, secrets=arguments.secrets, mode=arguments.mode, map_blobs=True)

	# get tag from the source database, adjust it, and copy it (with the defined IOV range) to the destination database

	print("Reading source Tag.")
	source_tag = source_connection.tag(name=arguments.input_tag)
	if source_tag == None:
		raise Exception("Source Tag doesn't exist.")

	# get all IOVs within the range [start, end]
	print("Reading source IOVs.")
	since_range = source_connection.range(arguments.start, arguments.end)
	source_iovs = source_tag.iovs(since=since_range).data()

	# get hashes of all IOVs contained in the Tag in the source database
	print("Reading source Payloads.")
	hashes = source_tag.iovs().get_members("payload_hash").data()
	payloads = source_connection.payload(hash=hashes)

	print("Writing to destination database...")

	# set end_of_validity to -1 because sqlite doesn't support long ints
	source_tag.end_of_validity = -1
	source_tag.name = arguments.dest_tag
	source_tag.modification_time = datetime.datetime.now()

	# create new iovs
	new_iovs = []
	for iov in source_iovs:
		new_iovs.append(dest_connection.models["iov"](iov.as_dicts(convert_timestamps=False), convert_timestamps=False))

	# write new tag to destination database
	print("Writing destination Tag.")
	if dest_connection.tag(name=arguments.dest_tag) != None:
		dest_connection.write_and_commit(source_tag)

	# write new iovs
	print("Writing IOVs to destination Tag.")
	for iov in new_iovs:
		if dest_connection.iov(tag_name=iov.tag_name, since=iov.since, insertion_time=iov.insertion_time) == None:
			dest_connection.write_and_commit(iov)

	# get payloads used by IOVs and copy those over
	print("Copying Payloads over.")
	for payload in payloads:
		if dest_connection.payload(hash=payload.hash) == None:
			dest_connection.write_and_commit(payload)

	print("Copy complete.")

def copy_global_tag(arguments):
	raise NotImplementedError("Copying Global Tags is currently not supported for this transition command-line interface for CondDBFW.")

	# set up connection
	source_connection = querying.connect(arguments.db, secrets=arguments.secrets, mode=arguments.mode, map_blobs=True)
	dest_connection = querying.connect(arguments.dest_db, secrets=arguments.secrets, mode=arguments.mode, map_blobs=True)

	# get CondDBFW Global Tag object
	global_tag = source_connection.global_tag(name=arguments.input_gt)
	if global_tag == None:
		raise Exception("Source Global Tag doesn't exist.")

	tag_names = global_tag.tags().get_members("tag_name").data()
	tags = source_connection.tag(name=tags)

	# copy global tag first
	global_tag.insertion_time = datetime.datetime.now()
	global_tag.validity = -1
	dest_connection.write_and_commit(global_tag)

	for tag in tags:
		# create temporary argument class
		class args(object):
			def __init__(self):
				self.input_tag = tag.name
				self.dest_tag = tag.name
				self.start = 1
				self.end = tag.latest_iov()+1
				for attribute in dir(arguments):
					self.__dict__[attribute] = getattr(arguments, attribute)

		copy_tag(args())

def parse_command_line(arguments):
	"""
	Assumes script name has been removed from the list of arguments.
	Hence, arguments[0] is the subcommand.
	"""
	top_level_parser = argparse.ArgumentParser(description="CondDBFW Command line tool")
	top_level_parser.add_argument("--db", type=str, required=False, default="frontier://FrontierProd/CMS_CONDITIONS")
	top_level_parser.add_argument("--mode", type=str, required=False, default="w")
	top_level_parser.add_argument("--secrets", type=str, required=False)
	top_level_parser.add_argument("--limit", type=int, required=False, default=10)

	subparser = top_level_parser.add_subparsers(title="Subcommands")

	list_parser = subparser.add_parser("list", description="Lists the Metadata objects contained within the given object.")
	list_parser.add_argument("--tag", required=False, help="List all IOVs in a Tag.")
	list_parser.add_argument("--gt", required=False, help="List all Global Tag Maps linked to a Global Tag.")
	list_parser.add_argument("--gts-for-tag", required=False, help="List all Global Tags that contain a Tag.")

	list_parser.set_defaults(func=list_object)

	diff_parser = subparser.add_parser("diff-tags", description="Gives the differences in payload hashes used by IOVs between Tags.")
	diff_parser.add_argument("--tag1", required=True, help="First Tag to use in the comparison.")
	diff_parser.add_argument("--tag2", required=True, help="Second Tag to use in the comparison.")

	diff_parser.set_defaults(func=diff_of_tags)

	gt_diff_parser = subparser.add_parser("diff-gts", description="Gives the differences in Global Tag Maps contained within Global Tag.")
	gt_diff_parser.add_argument("--gt1", required=True, help="First Global Tag to use in the comparison.")
	gt_diff_parser.add_argument("--gt2", required=True, help="Second Global Tag to use in the comparison.")

	gt_diff_parser.set_defaults(func=diff_of_gts)

	copy_tag_parser = subparser.add_parser("copy-tag", description="Copies a Tag with its IOVs and Payloads to a destination database."
														+ "\nFor copying to official databases, use cmsDbCondUpload (https://cms-conddb-dev.cern.ch/cmsDbCondUpload).")
	copy_tag_parser.add_argument("--dest-db", required=True, help="Database to copy the Tag and its IOVs to.")
	copy_tag_parser.add_argument("--input-tag", required=True, help="Tag to take data from in source database.")
	copy_tag_parser.add_argument("--dest-tag", required=True, help="Tag to copy input Tag to in the destination database.")
	copy_tag_parser.add_argument("--start", required=True, help="Since to start from.  If this is between two, the highest one is taken (no adjustments are made).")
	copy_tag_parser.add_argument("--end", required=True, help="Since to finidh at.  If this is between two, the lowest one is taken (no adjustments are made).")

	copy_tag_parser.set_defaults(func=copy_tag)

	parsed_arguments = top_level_parser.parse_args()

	print("Using database '%s'." % parsed_arguments.db)

	parsed_arguments.func(parsed_arguments)

if __name__ == "__main__":
	import sys
	parse_command_line(sys.argv[1:])