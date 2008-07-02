#!/bin/sh
# Returns number of copy workers running in each node
# gets list of node from host_list.cfg (1 hostname per line)
for i in `cat host_list.cfg`
do
	echo Processing $i:
	ssh $i ps -ef | grep Copy | wc -l;
	echo
done
