select dt_wheel_numbering.object_number as wheel,
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
       rownum as conn_id,(select max(iov_value_id) from dtreadoutmapping)
  from dt_wheel_numbering,dt_chamber_numbering,
       dt_superlayer_numbering,dt_layer_numbering,dt_cell_numbering,
       dt_wheel,dt_sector,dt_chamber,dt_superlayer,dt_layer,dt_cell,
       dt_cell_readout_map,dt_tdc_channel,dt_tdc,dt_rob,
       dt_rob_ros_connection,dt_ros_channel,dt_ros,dt_ddu,
       (select numbering_id from dt_numbering_master
         where numbering_version='ORCA_NUMBERING') dt_num,
       (select connection_id from dt_connection_master 
         where connection_type='DT_CELL_READOUT_CONNECTION'
           and mapping_version='FULL_CELL') rob_con,
       (select connection_id from dt_connection_master 
         where connection_type='DT_ROB_ROS_CONNECTION'
           and mapping_version='FULL_ROS') ros_con
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
    and dt_cell_readout_map.connection_id=rob_con.connection_id
    and dt_rob_ros_connection.connection_id=ros_con.connection_id
    and dt_wheel_numbering.numbering_id=dt_num.numbering_id
    and dt_chamber_numbering.numbering_id=dt_num.numbering_id
    and dt_superlayer_numbering.numbering_id=dt_num.numbering_id
    and dt_layer_numbering.numbering_id=dt_num.numbering_id
    and dt_cell_numbering.numbering_id=dt_num.numbering_id;  
