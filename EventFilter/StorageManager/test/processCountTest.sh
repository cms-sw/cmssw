# The sole purpose of this script is to test the ResourceMonitorCollection::processCount method.
# It is used by ResourceMonitorCollection_t.cpp, but is useless otherwise.

# If not called with arguments, spawn a slave process.
# Used to test that processCount only counts master processes
[[ $# == 0 ]] && ./processCountTest.sh slave &
sleep 5
