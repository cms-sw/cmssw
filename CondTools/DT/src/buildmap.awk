BEGIN {print "insert into dtreadoutmapping"                              \
             " (iov_value_id,cell_map_version,rob_map_version)"     \
             " values("NMAP",\047 \047,\047 \047);"}
      {print "insert into dtreadoutconnection" \
             " (connection_id,iov_value_id,ddu,ros,rob,tdc,channel,"     \
             "wheel,station,sector,superlayer,layer,cell)"               \
             " values ("NCON++","NMAP","$8","$9","$10","$11","$12","         \
             $2","$3","$4","$5","$6","$7");"}
 END {print "update dtreadoutmapping set cell_map_version"               \
            "=\047"CELL_MAP_VERSION"\047 where iov_value_id="NMAP";";    \
      print "update dtreadoutmapping set rob_map_version"               \
            "=\047" ROB_MAP_VERSION"\047 where iov_value_id="NMAP";";
      print "update pool_rss_containers" \
            " set number_of_written_objects=number_of_written_objects+1" \
            " where table_name=\047DTREADOUTMAPPING\047;"}
