/*
 *  DTReadOutMapping_payload_o2o()
 *
 *  DTReadOutMapping transform/transfer
 *  Parameters:  last_id:  The lower bounding IOV_VALUE_ID for objects to transfer
 */
CREATE OR REPLACE PROCEDURE DTReadOutMapping_payload_o2o (
  last_id IN NUMBER
)
AS

BEGIN
INSERT INTO dtreadoutmapping
            (iov_value_id,cell_map_version,rob_map_version)
     VALUES ( (SELECT max(iov_value_id)+1 FROM dtreadoutmapping),
              'CMSSW_CELL','CMSSW_ROS';
;

INSERT INTO dtreadoutconnection
            (wheel,station,sector,superlayer,layer,cell,
             ddu,ros,rob,tdc,channel,
             connection_id,iov_value_id)
SELECT dt_wheel_numbering.object_number,
       dt_chamber_numbering.station_number,
       dt_chamber_numbering.sector_number,
       dt_superlayer_numbering.object_number,
       dt_layer_numbering.object_number,
       dt_cell_numbering.object_number,
       dt_ddu.read_out_number,
       dt_ros.read_out_number,
       dt_ros_channel.read_out_number,
       dt_tdc.read_out_number,
       dt_tdc_channel.read_out_number,
       rownum,(SELECT max(iov_value_id) FROM dtreadoutmapping)
  FROM dt_wheel_numbering@cmsomds       dt_wheel_numbering,
       dt_chamber_numbering@cmsomds     dt_chamber_numbering,    
       dt_superlayer_numbering@cmsomds  dt_superlayer_numbering, 
       dt_layer_numbering@cmsomds       dt_layer_numbering,      
       dt_cell_numbering@cmsomds        dt_cell_numbering,       
       dt_wheel@cmsomds                 dt_wheel,                
       dt_sector@cmsomds                dt_sector,               
       dt_chamber@cmsomds               dt_chamber,              
       dt_superlayer@cmsomds            dt_superlayer,           
       dt_layer@cmsomds                 dt_layer,                
       dt_cell@cmsomds                  dt_cell,                 
       dt_cell_readout_map@cmsomds      dt_cell_readout_map,     
       dt_tdc_channel@cmsomds           dt_tdc_channel,          
       dt_tdc@cmsomds                   dt_tdc,                  
       dt_rob@cmsomds                   dt_rob,                  
       dt_rob_ros_connection@cmsomds    dt_rob_ros_connection,   
       dt_ros_channel@cmsomds           dt_ros_channel,          
       dt_ros@cmsomds                   dt_ros,                  
       dt_ddu@cmsomds                   dt_ddu,                  
       (SELECT numbering_id FROM dt_numbering_master@cmsomds
         WHERE numbering_version='CMSSW_NUMBERING') dt_num,
       (SELECT connection_id FROM dt_connection_master@cmsomds
         WHERE connection_type='DT_CELL_READOUT_CONNECTION'
           AND
        mapping_version='CMSSW_CELL') rob_con,
       (SELECT connection_id FROM dt_connection_master@cmsomds
         WHERE connection_type='DT_ROB_ROS_CONNECTION'
           AND mapping_version='CMSSW_ROS') ros_con
  WHERE dt_cell_readout_map.cell_id=dt_cell.cell_id
    AND dt_cell.layer_id=dt_layer.layer_id
    AND dt_layer.sl_id=dt_superlayer.sl_id
    AND dt_superlayer.chamber_id=dt_chamber.chamber_id
    AND dt_chamber.sector_id=dt_sector.sector_id
    AND dt_sector.wheel_id=dt_wheel.wheel_id
    AND dt_wheel_numbering.wheel_id=dt_wheel.wheel_id
    AND dt_chamber_numbering.chamber_id=dt_chamber.chamber_id
    AND dt_superlayer_numbering.sl_id=dt_superlayer.sl_id
    AND dt_layer_numbering.layer_id=dt_layer.layer_id
    AND dt_cell_numbering.cell_id=dt_cell.cell_id
    AND dt_cell_readout_map.tdc_channel_id=dt_tdc_channel.tdc_channel_id
    AND dt_tdc_channel.tdc_id=dt_tdc.tdc_id
    AND dt_tdc.rob_id=dt_rob.rob_id
    AND dt_rob.rob_id=dt_rob_ros_connection.rob_id
    AND dt_ros_channel.ros_channel_id=dt_rob_ros_connection.ros_channel_id
    AND dt_ros_channel.ros_id=dt_ros.ros_id
    AND dt_ros.ddu_id=dt_ddu.ddu_id
    AND dt_cell_readout_map.connection_id=rob_con.connection_id
    AND dt_rob_ros_connection.connection_id=ros_con.connection_id
    AND dt_wheel_numbering.numbering_id=dt_num.numbering_id
    AND dt_chamber_numbering.numbering_id=dt_num.numbering_id
    AND dt_superlayer_numbering.numbering_id=dt_num.numbering_id
    AND dt_layer_numbering.numbering_id=dt_num.numbering_id
    AND dt_cell_numbering.numbering_id=dt_num.numbering_id;  
;
END;
/
show errors;