echo "===== Running test of friendly names ======"
edmPluginDump | grep Wrapper | awk -F/ '{print "\""$2"\""}' | xargs edmToFriendlyClassName || die ">>>>> test of friendly names failed <<<<<" $?
