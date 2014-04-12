echo "===== Running test of friendly names ======"
edmPluginDump | grep Wrapper | awk -F/ '{print "\""$2"\""}' | xargs edmToFriendlyClassName >/dev/null 2>&1 || die ">>>>> test of friendly names failed <<<<<" $?
