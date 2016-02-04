/*
 *  DTReadOutMapping_payload_o2o()
 *
 *  DTReadOutMapping transform/transfer
 *  Parameters:  last_id:  The lower bounding IOV_VALUE_ID for objects to transfer
 *
 *  Note:  Uses sequence on ORCON for IOV_VALUE_ID
 *         Transfers 1 object per call
 */

/* A sequence for the DTReadoutMapping object IOV_VALUE_ID */
DROP SEQUENCE dtread_id_sq;
CREATE SEQUENCE dtread_id_sq
START WITH 1
INCREMENT BY 1
;

CREATE OR REPLACE PROCEDURE DTReadOutMapping_payload_o2o (
  last_id IN NUMBER
)
AS

BEGIN
INSERT INTO dtreadoutmapping
            (iov_value_id,cell_map_version,rob_map_version,time)
     VALUES ( dtread_id_sq.NextVal, 'CMSSW_CELL','CMSSW_ROS', NULL )
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
       rownum, dtread_id_sq.CurrVal
  FROM dt_wheel_numbering@omds       dt_wheel_numbering,
       dt_chamber_numbering@omds     dt_chamber_numbering,    
       dt_superlayer_numbering@omds  dt_superlayer_numbering, 
       dt_layer_numbering@omds       dt_layer_numbering,      
       dt_cell_numbering@omds        dt_cell_numbering,       
       dt_wheel@omds                 dt_wheel,                
       dt_sector@omds                dt_sector,               
       dt_chamber@omds               dt_chamber,              
       dt_superlayer@omds            dt_superlayer,           
       dt_layer@omds                 dt_layer,                
       dt_cell@omds                  dt_cell,                 
       dt_cell_readout_map@omds      dt_cell_readout_map,     
       dt_tdc_channel@omds           dt_tdc_channel,          
       dt_tdc@omds                   dt_tdc,                  
       dt_rob@omds                   dt_rob,                  
       dt_rob_ros_connection@omds    dt_rob_ros_connection,   
       dt_ros_channel@omds           dt_ros_channel,          
       dt_ros@omds                   dt_ros,                  
       dt_ddu@omds                   dt_ddu,                  
       (SELECT numbering_id FROM dt_numbering_master@omds
         WHERE numbering_version='CMSSW_NUMBERING') dt_num,
       (SELECT connection_id FROM dt_connection_master@omds
         WHERE connection_type='DT_CELL_READOUT_CONNECTION'
           AND mapping_version='CMSSW_CELL') rob_con,
       (SELECT connection_id FROM dt_connection_master@omds
         WHERE connection_type='DT_ROB_ROS_CONNECTION'
           AND mapping_version='CMSSW_ROS') ros_con,
        dual /* No idea why this is needed, but it is (sequence?) */
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
    AND dt_cell_numbering.numbering_id=dt_num.numbering_id
;

END;
/
show errors;
