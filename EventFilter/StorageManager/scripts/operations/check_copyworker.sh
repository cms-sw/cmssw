#!/bin/sh
# $Id: check_copyworker.sh,v 1.2 2008/07/03 10:46:58 loizides Exp $

# Returns number of copy workers running in each node
# gets list of node from host_list.cfg (1 hostname per line)

for i in `cat host_list.cfg`
do
  echo Processing $i:
  ssh $i ps -ef | grep Copy | wc -l;
  echo
done
