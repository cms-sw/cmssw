#!/bin/csh

setenv  CELL_MAP_VERSION $1
setenv   ROB_MAP_VERSION $2
setenv NUMBERING_VERSION $3

#setenv DB_SRC drifttube@cmsomds/daqcms

setenv DB_SRC CMS_VAL_DT_POOL_OWNER@cms_val_lb.cern.ch/val_dt_own_1031
setenv DB_END CMS_VAL_DT_POOL_WRITER@cms_val_lb.cern.ch/val_dt_wri_1031

#setenv DB_SRC cms_dt_writer@devdb/daqcms123
#setenv DB_END cms_dt_writer@devdb/daqcms123

setenv TMPFILE /tmp/buildscript`date +%s`

# count maps in destination db

sqlplus ${DB_END} << EOF | grep ENTRY | grep -vi mapping | awk -v NMAX=0 '($2>=NMAX){NMAX=1+$2} END {print "setenv NMAP "NMAX}' > ${TMPFILE}
select 'ENTRY',iov_value_id from dtreadoutmapping;
EOF

# retrieve original maps identifiers

sqlplus ${DB_SRC} << EOF | grep ENTRY | grep -vi connection | awk -v CELL_MAP_VERSION=${CELL_MAP_VERSION} '{print "setenv CELL_MAP_ID "$2}' >> ${TMPFILE}
select 'ENTRY',connection_id from dt_connection_master where connection_type='DT_CELL_READOUT_CONNECTION' and mapping_version='${CELL_MAP_VERSION}';
EOF

sqlplus ${DB_SRC} << EOF | grep ENTRY | grep -vi connection | awk -v ROB_MAP_VERSION=${ROB_MAP_VERSION} '{print "setenv ROB_MAP_ID "$2}' >> ${TMPFILE}
select 'ENTRY',connection_id from dt_connection_master where connection_type='DT_ROB_ROS_CONNECTION' and mapping_version='${ROB_MAP_VERSION}';
EOF

sqlplus ${DB_SRC} << EOF | grep ENTRY | grep -vi numbering | awk -v NUMBERING_VERSION=${NUMBERING_VERSION} '{print "setenv NUMBERING_ID "$2}' >> ${TMPFILE}
select 'ENTRY',numbering_id from dt_numbering_master where numbering_version='${NUMBERING_VERSION}';
EOF

source ${TMPFILE}
rm -f ${TMPFILE}

# build offline map

echo ${NMAP}" "${CELL_MAP_ID}" "${ROB_MAP_ID}

sqlplus ${DB_SRC} << EOF | grep ENTRY | grep -vi wheel | awk -v NMAP=${NMAP} -v CELL_MAP_VERSION=${CELL_MAP_VERSION} -v ROB_MAP_VERSION=${ROB_MAP_VERSION} -v NCON=0 -f buildmap.awk | sqlplus ${DB_END}

set line 200
set pagesize 2
select 'ENTRY',
       dt_wheel_numbering.object_number as wheel,
       dt_chamber_numbering.station_number as station,
       dt_chamber_numbering.sector_number as sector,
       dt_superlayer_numbering.object_number as sl,
       dt_layer_numbering.object_number as layer,
       dt_cell_numbering.object_number as cell,
       dt_ddu.read_out_number as ddu,
       dt_ros.read_out_number as ros,
       dt_ros_channel.read_out_number as rob,
       dt_tdc.read_out_number as tdc,
       dt_tdc_channel.read_out_number as channel,
       dt_cell_readout_map.connection_id,
       dt_rob_ros_connection.connection_id
  from dt_wheel_numbering,dt_chamber_numbering,
       dt_superlayer_numbering,dt_layer_numbering,dt_cell_numbering,
       dt_wheel,dt_sector,dt_chamber,dt_superlayer,dt_layer,dt_cell,
       dt_cell_readout_map,dt_tdc_channel,dt_tdc,dt_rob,
       dt_rob_ros_connection,dt_ros_channel,dt_ros,dt_ddu
  where dt_cell_readout_map.cell_id=dt_cell.cell_id
    and dt_cell.layer_id=dt_layer.layer_id
    and dt_layer.sl_id=dt_superlayer.sl_id
    and dt_superlayer.chamber_id=dt_chamber.chamber_id
    and dt_chamber.sector_id=dt_sector.sector_id
    and dt_sector.wheel_id=dt_wheel.wheel_id
    and dt_wheel_numbering.wheel_id=dt_wheel.wheel_id
    and dt_chamber_numbering.chamber_id=dt_chamber.chamber_id
    and dt_superlayer_numbering.sl_id=dt_superlayer.sl_id
    and dt_layer_numbering.layer_id=dt_layer.layer_id
    and dt_cell_numbering.cell_id=dt_cell.cell_id
    and dt_cell_readout_map.tdc_channel_id=dt_tdc_channel.tdc_channel_id
    and dt_tdc_channel.tdc_id=dt_tdc.tdc_id
    and dt_tdc.rob_id=dt_rob.rob_id
    and dt_rob.rob_id=dt_rob_ros_connection.rob_id
    and dt_ros_channel.ros_channel_id=dt_rob_ros_connection.ros_channel_id
    and dt_ros_channel.ros_id=dt_ros.ros_id
    and dt_ros.ddu_id=dt_ddu.ddu_id
    and dt_cell_readout_map.connection_id=${CELL_MAP_ID}
    and dt_rob_ros_connection.connection_id=${ROB_MAP_ID}
    and dt_wheel_numbering.numbering_id=${NUMBERING_ID}
    and dt_chamber_numbering.numbering_id=${NUMBERING_ID}
    and dt_superlayer_numbering.numbering_id=${NUMBERING_ID}
    and dt_layer_numbering.numbering_id=${NUMBERING_ID}
    and dt_cell_numbering.numbering_id=${NUMBERING_ID};
quit;
EOF

