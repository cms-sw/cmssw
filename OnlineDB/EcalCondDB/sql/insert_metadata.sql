/* INSERT something */

INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'CHANNELVIEW',1,0,'DEF','channel maps definition in the ECAL','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'VIEWDESCRIPTION',4,0,'DEF','list of the various channel numbering schemes','none','none' );                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'O2O_LOG',5,0,'DEF','log of the online to offline transport','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'MON_RUN_IOV',3,0,'DQM','DQM Interval of validity with subrun','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'LOCATION_DEF',1,0,'DEF','Definition of the various sites where ECAL takes data','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'RUN_TYPE_DEF',1,0,'DEF','Definition of the run types taken by ECAL','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'RUN_TAG',6,0,'RUN_CONTROL','Definition of the run tags','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'RUN_IOV',3,0,'RUN_CONTROL','list of runs taken','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'RUN_DAT',2,1,'RUN_CONTROL','Run data and properties','by Sm or entire ECAL','EB_supermodule',1,1);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'MON_VERSION_DEF',1,0,'DEF','Definition of monitoring versions','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'MON_RUN_TAG',6,0,'DEF','Monitoring tags','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'MON_RUN_OUTCOME_DEF',1,0,'DEF','Monitoring outcome def','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'MON_RUN_DAT',2,0,'DEF','Monitoring run list','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'MON_CRYSTAL_STATUS_DEF',1,0,'DEF','definition of possible crystal status','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_CRYSTAL_STATUS_DAT',2,3,'DQM','crystal status','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'MON_PN_STATUS_DEF',1,0,'DEF','PN status definition','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_STATUS_DAT',2,2,'DQM','PN status','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_CRYSTAL_CONSISTENCY_DAT',2,6,'DQM','crystal consistency','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_TT_CONSISTENCY_DAT',2,7,'DQM','Trigger tower consistency','TT numbering','EB_trigger_tower',17,4);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_OCCUPANCY_DAT',2,3,'DQM','crystal occupancy','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PEDESTALS_DAT',2,7,'DQM','crystal pedestals','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PEDESTALS_ONLINE_DAT',2,3,'DQM','crystal pedestals from pre-samples','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PEDESTAL_OFFSETS_DAT',2,4,'DQM','crystal optimal DAC for pedestals','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_TEST_PULSE_DAT',2,7,'DQM','test-pulse run results','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PULSE_SHAPE_DAT',2,30,'DQM','average channel pulse shape values','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_SHAPE_QUALITY_DAT',2,1,'DQM','channel pulse shape quality','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_DELAYS_TT_DAT',2,3,'DQM','channel delays for optimal synchronization','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_BLUE_DAT',2,9,'DQM','PN analysis in blue laser runs','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_GREEN_DAT',2,9,'DQM','PN analysis in green laser runs','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_RED_DAT',2,9,'DQM','PN analysis in red laser runs','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_IRED_DAT',2,9,'DQM','PN analysis in infra-red laser runs','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_PED_DAT',2,5,'DQM','PN pedestals','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_MGPA_DAT',2,10,'DQM','PN test-pulse','PN numbering', 'EB_LM_PN',5,2);  

INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_LASER_BLUE_DAT',2,5,'DQM','Blue Laser analysis','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_LASER_GREEN_DAT',2,5,'DQM','Green Laser analysis','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_LASER_RED_DAT',2,5,'DQM','Red Laser analysis','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_LASER_IRED_DAT',2,5,'DQM','Infra-red Laser analysis','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_MEM_CH_CONSISTENCY_DAT',2,6,'DQM','Monitoring electronics analysis','mem channel numbering','EB_mem_channel',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_MEM_TT_CONSISTENCY_DAT',2,7,'DQM','Monitoring electronics pseudo-TT analysis','mem TT channel numbering','EB_mem_TT',1,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'DCU_TAG',6,0,'CCS_Supervisor','Definition of the run tags for DCU runs','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'DCU_IOV',3,0,'CCS_Supervisor','List of DCU runs (Temp,IDark,Voltages)','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_CAPSULE_TEMP_DAT',2,1,'CCS_Supervisor','Capsule Calibrated Temperatures','Temperature capsule numbering','EB_T_capsule',17,10);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_CAPSULE_TEMP_RAW_DAT',2,1,'CCS_Supervisor','Capsule Uncalibrated Temperatures','Temperature capsule numbering','EB_T_capsule',17,10);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_IDARK_DAT',2,1,'CCS_Supervisor','APD Dark Currents','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_IDARK_PED_DAT',2,1,'CCS_Supervisor','APD Dark Currents DCU Pedestals','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_VFE_TEMP_DAT',2,1,'CCS_Supervisor','VFE card temperatures','VFE numbering','EB_VFE',17,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_LVR_TEMPS_DAT',2,3,'CCS_Supervisor','LVR chip temperatures','LVR numbering','EB_LVRB_DCU',17,4);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_LVRB_TEMPS_DAT',2,3,'CCS_Supervisor','LVR board temperatures','LVR numbering','EB_LVRB_T_sensor',17,4);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_LVR_VOLTAGES_DAT',2,11,'CCS_Supervisor','LVR measured voltages','LVR numbering','EB_trigger_tower',17,4);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'LMF_RUN_TAG',6,0,'LaserMonitoringFarm','Definition of the run tags for LMF analysis','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME) values (COND_TABLE_SQ.NextVal, 'LMF_RUN_IOV',3,0,'LaserMonitoringFarm','LMF Interval of validity with subrun','none','none');                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_LASER_BLUE_RAW_DAT',2,2,'LaserMonitoringFarm','Blue Laser raw analysis','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_LASER_IRED_RAW_DAT',2,2,'LaserMonitoringFarm','Infra-red Laser raw analysis','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_LASER_BLUE_NORM_DAT',2,6,'LaserMonitoringFarm','Blue Laser APD/PN results','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_LASER_IRED_NORM_DAT',2,6,'LaserMonitoringFarm','Infra-red Laser APD/PN results','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_LASER_BLUE_COEFF_DAT',2,2,'LaserMonitoringFarm','Blue Laser compensation coefficients','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_LASER_IRED_COEFF_DAT',2,2,'LaserMonitoringFarm','Infra-red Laser compensation coefficients','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_LASER_BLUE_SHAPE_DAT',2,4,'LaserMonitoringFarm','Blue Laser Pulse Shape Parameters','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_LASER_IRED_SHAPE_DAT',2,4,'LaserMonitoringFarm','Blue Laser Pulse Shape Parameters','crystal numbering','EB_crystal_number',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_PN_BLUE_DAT',2,2,'LaserMonitoringFarm','Blue Laser PN analysis','PN numbering', 'EB_LM_PN',5,10);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_PN_IRED_DAT',2,2,'LaserMonitoringFarm','Infra-red Laser PN analysis','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_PN_TEST_PULSE_DAT',2,2,'LaserMonitoringFarm','PN test-pulse analysis','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'LMF_PN_CONFIG_DAT',2,5,'LaserMonitoringFarm','indication of PN used in LASER normalization','PN numbering', 'EB_LM_PN',5,2);                                                                                  

/* RUN_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 9,'NUM_EVENTS',1,'NUMBER','Number of Events in a Run','Number of Events'); 

/* MON RUN_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'NUM_EVENTS',1,'NUMBER','Number of Events in a Run','Number of Events'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'RUN_OUTCOME_ID',1,'NUMBER','Number of Events in a Run','Number of Events'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'ROOTFILE_NAME',0,'STRING','Root file name','none');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'TASK_LIST',0,'INT','Task list','none');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'TASK_OUTCOME',1,'INT','Task outcome','none');

/* MON_CRYSTAL_STATUS_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,15, 'STATUS_G1',1,'INT','Channel Status Gain1','Channel Status Gain1');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,15, 'STATUS_G6',1,'INT','Channel Status Gain6','Channel Status Gain6');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,15, 'STATUS_G12',1,'INT','Channel Status Gain12','Channel Status Gain12');

/* MON_PN_STATUS_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 17 , 'STATUS_G1',1,'INT','Channel Status Gain1','Channel Status Gain1');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 17 , 'STATUS_G16',1,'INT','Channel Status Gain16','Channel Status Gain16');

/* MON_CRYSTAL_CONSISTENCY_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 18, 'PROCESSED_EVENTS',1,'INT','Number of processed events','Number of processed events');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 18, 'PROBLEMATIC_EVENTS',1,'INT','Number of problematic events','Number of problematic events');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 18, 'PROBLEMS_ID',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 18, 'PROBLEMS_GAIN_ZERO',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 18, 'PROBLEMS_GAIN_SWITCH',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 18, 'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_TT_CONSISTENCY_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 19, 'PROCESSED_EVENTS',1,'INT','Number of processed events','Number of processed events');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMATIC_EVENTS',1,'INT','Number of problematic events','Number of problematic events');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMS_ID',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMS_SIZE',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMS_LV1',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMS_BUNCH_X',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 19, 'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_OCCUPANCY_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 20 ,'EVENTS_OVER_LOW_THRESHOLD',1,'INT','Number of events above low energy threshold','Number of events above low threshold');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 20 ,'EVENTS_OVER_HIGH_THRESHOLD',1,'INT','Number of events above high energy threshold','Number of events above high threshold');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 20 ,'AVG_ENERGY',1,'FLOAT','Average energy per channel','Average energy per channel (ADC Counts)');


/* MON_PEDESTALS_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 21,'PED_MEAN_G1',1,'FLOAT','Pedestal gain 1','Pedestal Gain1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 21,'PED_MEAN_G6',1,'FLOAT','Pedestal gain 6','Pedestal Gain6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 21,'PED_MEAN_G12',1,'FLOAT','Pedestal gain 12','Pedestal Gain12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 21,'PED_RMS_G1',1,'FLOAT','Pedestal RMS gain 1','RMS Pedestal Gain1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 21,'PED_RMS_G6',1,'FLOAT','Pedestal RMS gain 6','RMS Pedestal Gain6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 21,'PED_RMS_G12',1,'FLOAT','Pedestal RMS gain 12','RMS Pedestal Gain12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 21, 'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_PEDESTALS_ONLINE_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 22,'ADC_MEAN_G12',1,'FLOAT','Online Pedestal gain 12','Online Pedestal Gain12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 22,'ADC_RMS_G12',1,'FLOAT','Online Pedestal RMS gain 12','RMS Online Pedestal Gain12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 22,'TASK_STATUS',1,'INT','Task status','Task status');


/* MON_PEDESTAL_OFFSETS_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,23 ,'DAC_G1',1,'INT','DAC for Pedestal gain 1','DAC for Pedestal Gain1');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 23,'DAC_G6',1,'INT','DAC for Pedestal gain 6','DAC for Pedestal Gain6');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 23,'DAC_G12',1,'INT','DAC for Pedestal gain 12','DAC for Pedestal Gain12'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 23, 'TASK_STATUS',1,'INT','Task status','Task status');


/* MON_TEST_PULSE_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 24,'ADC_MEAN_G1',1,'FLOAT','Mean Test-pulse Gain 1','Test-pulse Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 24,'ADC_RMS_G1',1,'FLOAT','RMS Test-pulse Gain 1','RMS Test-pulse Gain 1(ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 24,'ADC_MEAN_G6',1,'FLOAT','Mean Test-pulse Gain 6','Mean Test-pulse Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 24,'ADC_RMS_G6',1,'FLOAT','RMS Test-pulse Gain 6','RMS Test-pulse Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 24,'ADC_MEAN_G12',1,'FLOAT','Mean Test-pulse Gain 12','Mean Test-pulse Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 24,'ADC_RMS_G12',1,'FLOAT','RMS Test-pulse Gain 12','RMS Test-pulse Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 24, 'TASK_STATUS',1,'INT','Task status','Task status');


 
/*  MON_PULSE_SHAPE_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_01',1,'FLOAT','Mean height sample 01 Gain 1','Mean height sample 01 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_02',1,'FLOAT','Mean height sample 02 Gain 1','Mean height sample 02 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_03',1,'FLOAT','Mean height sample 03 Gain 1','Mean height sample 03 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_04',1,'FLOAT','Mean height sample 04 Gain 1','Mean height sample 04 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_05',1,'FLOAT','Mean height sample 05 Gain 1','Mean height sample 05 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_06',1,'FLOAT','Mean height sample 06 Gain 1','Mean height sample 06 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_07',1,'FLOAT','Mean height sample 07 Gain 1','Mean height sample 07 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_08',1,'FLOAT','Mean height sample 08 Gain 1','Mean height sample 08 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_09',1,'FLOAT','Mean height sample 09 Gain 1','Mean height sample 09 Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_10',1,'FLOAT','Mean height sample 10 Gain 1','Mean height sample 10 Gain 1 (ADC Counts)');
/* samples at gain 6 */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_01',1,'FLOAT','Mean height sample 01 Gain 6','Mean height sample 01 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_02',1,'FLOAT','Mean height sample 02 Gain 6','Mean height sample 02 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_03',1,'FLOAT','Mean height sample 03 Gain 6','Mean height sample 03 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_04',1,'FLOAT','Mean height sample 04 Gain 6','Mean height sample 04 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_05',1,'FLOAT','Mean height sample 05 Gain 6','Mean height sample 05 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_06',1,'FLOAT','Mean height sample 06 Gain 6','Mean height sample 06 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_07',1,'FLOAT','Mean height sample 07 Gain 6','Mean height sample 07 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_08',1,'FLOAT','Mean height sample 08 Gain 6','Mean height sample 08 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_09',1,'FLOAT','Mean height sample 09 Gain 6','Mean height sample 09 Gain 6 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_10',1,'FLOAT','Mean height sample 10 Gain 6','Mean height sample 10 Gain 6 (ADC Counts)');
/* samples at gain 12 */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_01',1,'FLOAT','Mean height sample 01 Gain 12','Mean height sample 01 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_02',1,'FLOAT','Mean height sample 02 Gain 12','Mean height sample 02 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_03',1,'FLOAT','Mean height sample 03 Gain 12','Mean height sample 03 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_04',1,'FLOAT','Mean height sample 04 Gain 12','Mean height sample 04 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_05',1,'FLOAT','Mean height sample 05 Gain 12','Mean height sample 05 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_06',1,'FLOAT','Mean height sample 06 Gain 12','Mean height sample 06 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_07',1,'FLOAT','Mean height sample 07 Gain 12','Mean height sample 07 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_08',1,'FLOAT','Mean height sample 08 Gain 12','Mean height sample 08 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_09',1,'FLOAT','Mean height sample 09 Gain 12','Mean height sample 09 Gain 12 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_10',1,'FLOAT','Mean height sample 10 Gain 12','Mean height sample 10 Gain 12 (ADC Counts)');


/* MON_SHAPE_QUALITY_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 26,'AVG_CHI2',1,'FLOAT','Average chi2 of the pulse fit','Fit Average Chi2');

/* MON_DELAYS_TT_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 27,'DELAY_MEAN',1,'FLOAT','Mean Delay per Trigger Tower','Mean TT Delay (ns)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 27,'DELAY_RMS',1,'FLOAT','RMS Delay per Trigger Tower','RMS TT Delay (ns)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 27, 'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_PN_BLUE_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Blue Laser PN Height Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Blue Laser PN RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Blue Laser PN Height Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Blue Laser PN RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Blue Laser PN Pedestal Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Blue Laser PN Pedestal RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Blue Laser PN Pedestal Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Blue Laser PN Pedestal RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 28, 'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_PN_GREEN_DAT */ 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Green Laser PN Height Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Green Laser PN RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Green Laser PN Height Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Green Laser PN RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Green Laser PN Pedestal Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Green Laser PN Pedestal RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Green Laser PN Pedestal Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Green Laser PN Pedestal RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 29, 'TASK_STATUS',1,'INT','Task status','Task status');


/* MON_PN_RED_DAT    */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Red Laser PN Height Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Red Laser PN RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Red Laser PN Height Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Red Laser PN RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Red Laser PN Pedestal Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Red Laser PN Pedestal RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Red Laser PN Pedestal Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Red Laser PN Pedestal RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 30, 'TASK_STATUS',1,'INT','Task status','Task status');



/* MON_PN_IRED_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Infra-Red Laser PN Height Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Infra-Red Laser PN RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Infra-Red Laser PN Height Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Infra-Red Laser PN RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Infra-Red Laser PN Pedestal Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Infra-Red Laser PN Pedestal RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Infra-Red Laser PN Pedestal Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Infra-Red Laser PN Pedestal RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 31,'TASK_STATUS',1,'INT','Task status','Task status');




/* MON_PN_PED_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,32,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',   'Mean PN Pedestal Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,32,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1','Mean PN Pedestal RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,32,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean PN Pedestal Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,32,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean PN Pedestal RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,32,'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_PN_MGPA_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Test-Pulse PN Height Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Test-Pulse PN RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Test-Pulse PN Height Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Test-Pulse PN RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean PN Pedestal from Test-Pulse runs Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean PN Pedestal from Test-Pulse runs RMS Gain 1 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean PN Pedestal from Test-Pulse runs Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean PN Pedestal from Test-Pulse runs RMS Gain 16 (ADC Counts)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 33,'TASK_STATUS',1,'INT','Task status','Task status');


/* MON_LASER_BLUE_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 34,'APD_MEAN',1,'FLOAT','Mean height',       'Mean Channel Height for Blue Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 34,'APD_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Blue Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 34,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PN',       'Mean APD/PN for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 34,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PN',       'RMS APD/PN for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 34,'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_LASER_GREEN_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 35,'APD_MEAN',1,'FLOAT','Mean height',       'Mean Channel Height for Green Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 35,'APD_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Green Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 35,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PN',       'Mean APD/PN for Green Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 35,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PN',       'RMS APD/PN for Green Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 35,'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_LASER_RED_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 36,'APD_MEAN',1,'FLOAT','Mean height',       'Mean Channel Height for Red Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 36,'APD_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Red Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 36,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PN',       'Mean APD/PN for Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 36,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PN',       'RMS APD/PN for Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 36,'TASK_STATUS',1,'INT','Task status','Task status');

/* MON_LASER_IRED_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 37,'APD_MEAN',1,'FLOAT','Mean height',       'Mean Channel Height for Infra-Red Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 37,'APD_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Infra-Red Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 37,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PN',       'Mean APD/PN for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 37,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PN',       'RMS APD/PN for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 37,'TASK_STATUS',1,'INT','Task status','Task status');


/* MON_MEM_CH_CONSISTENCY_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 38, 'PROCESSED_EVENTS',1,'INT','Number of processed events','Number of processed events');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 38, 'PROBLEMATIC_EVENTS',1,'INT','Number of problematic events','Number of problematic events');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 38, 'PROBLEMS_ID',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 38, 'PROBLEMS_GAIN_ZERO',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 38, 'PROBLEMS_GAIN_SWITCH',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 38, 'TASK_STATUS',1,'INT','Task status','Task status');


/* MON_MEM_TT_CONSISTENCY_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 39, 'PROCESSED_EVENTS',1,'INT','Number of processed events','Number of processed events');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMATIC_EVENTS',1,'INT','Number of problematic events','Number of problematic events');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMS_ID',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMS_SIZE',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMS_LV1',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMS_BUNCH_X',1,'INT','????','????');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 39, 'TASK_STATUS',1,'INT','Task status','Task status');


/* DCU_CAPSULE_TEMP_DAT     */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,42, 'CAPS_TEMP',1,'FLOAT','Capsule Temperature',       'Capsule Temperature (C)'); 

/* DCU_CAPSULE_TEMP_RAW_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,43, 'CAPS_TEMP_ADC',1,'FLOAT','Raw Capsule Temperature',       'Uncalibrated Capsule Temperature (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,43, 'CAPS_TEMP_RMS',1,'FLOAT','Raw Capsule Temperature RMS',       'Uncalibrated Capsule Temperature RMS (ADC Counts)'); 

/* DCU_IDARK_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,44, 'APD_IDARK',1,'FLOAT','Capsule Dark Current',       'Capsule Dark Current (microAmpere)'); 

/* DCU_IDARK_PED_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,45,'PED',1,'FLOAT','Capsule Dark Current DCU Pedestal',       'Capsule Dark Current DCU Pedestal (ADC Counts)'); 

/* DCU_VFE_TEMP_DAT   */ 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,46,'VFE_TEMP',1,'FLOAT','VFE Temperature',       'VFE Temperature (C)');   

/* DCU_LVR_TEMPS_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,47 ,'T1',1,'FLOAT','LVR Chip Temperature Sensor 1',       'LVR Chip Temperature on Sensor 1 (C)');  

INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,47 ,'T2',1,'FLOAT','LVR Chip Temperature Sensor 2',       'LVR Chip Temperature on Sensor 2 (C)'); 
 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,47 ,'T3',1,'FLOAT','LVR Chip Temperature Sensor 3',       'LVR Chip Temperature on Sensor 3 (C)');  
 
/* DCU_LVRB_TEMPS_DAT     */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,48 ,'T1',1,'FLOAT','LVRB Temperatures',       'LVRB Temperature Sensor 1 (C)');  
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,48 ,'T2',1,'FLOAT','LVRB Temperatures',       'LVRB Temperature Sensor 2 (C)');  
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,48 ,'T3',1,'FLOAT','LVRB Temperatures',       'LVRB Temperature Sensor 3 (C)');  


/* DCU_LVR_VOLTAGES_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'VFE1_A',1,'FLOAT','VFE 1 Analog Voltage',       'VFE 1 Analog Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'VFE2_A',1,'FLOAT','VFE 2 Analog Voltage',       'VFE 2 Analog Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'VFE3_A',1,'FLOAT','VFE 3 Analog Voltage',       'VFE 3 Analog Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'VFE4_A',1,'FLOAT','VFE 4 Analog Voltage',       'VFE 4 Analog Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'VFE5_A',1,'FLOAT','VFE 5 Analog Voltage',       'VFE 5 Analog Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'VCC',1,'FLOAT','VCC',       'VCC (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'VFE4_5_D',1,'FLOAT','VFE 4 and 5 Digital Voltage',       'VFE 4 and 5 Digital Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'VFE1_2_3_D',1,'FLOAT','VFE 1 2 and 3 Digital Voltage',       'VFE 1 2 and 3 Digital Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'BUFFER',1,'FLOAT','Buffer Voltage',       'Buffer Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'FENIX',1,'FLOAT','Fenix Voltage',       'Fenix Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'V43_A',1,'FLOAT','Input 4.3 V Analog Voltage',       'Input 4.3 V Analog Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'OCM',1,'FLOAT','OCM',       'OCM Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'GOH',1,'FLOAT','GOH',       'GOH Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'INH',1,'FLOAT','Inhibit',       'Inhibit Voltage (V)');   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal,49,'V43_D',1,'FLOAT','Input 4.3 V Digital Voltage',       'Input 4.3 V Digital Voltage (V)');   




/* LMF_LASER_BLUE_RAW_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 52,'APD_PEAK',1,'FLOAT','Mean height',       'Mean Channel Height for Blue Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 52,'APD_ERR',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Blue Laser runs (ADC Counts)'); 

/* LMF_LASER_IRED_RAW_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 53,'APD_PEAK',1,'FLOAT','Mean height',       'Mean Channel Height for Infra-Red Laser runs (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 53,'APD_ERR',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Infra-Red Laser runs (ADC Counts)'); 

/* LMF_LASER_BLUE_NORM_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PNA_MEAN',1,'FLOAT','Mean APD/PN 0',       'Mean APD/PN0 for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PNA_RMS',1,'FLOAT','RMS APD/PN 0',       'RMS APD/PN0 for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PNB_MEAN',1,'FLOAT','Mean APD/PN 1',       'Mean APD/PN1 for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PNB_RMS',1,'FLOAT','RMS APD/PN 1',       'RMS APD/PN1 for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PNmean',       'Mean APD/PNmean for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PNmean',       'RMS APD/PNmean for Blue Laser runs'); 



/* LMF_LASER_IRED_NORM_DAT    */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PNA_MEAN',1,'FLOAT','Mean APD/PN 0',       'Mean APD/PN0 for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PNA_RMS',1,'FLOAT','RMS APD/PN 0',       'RMS APD/PN0 for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PNB_MEAN',1,'FLOAT','Mean APD/PN 1',       'Mean APD/PN1 for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PNB_RMS',1,'FLOAT','RMS APD/PN 1',       'RMS APD/PN1 for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PNmean',       'Mean APD/PNmean for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PNmean',       'RMS APD/PNmean for Infra-Red Laser runs'); 


/* LMF_LASER_BLUE_COEFF_DAT    */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 56,'XPORT_COEFF',1,'FLOAT','Compensation Coefficient',       'Compensation Coefficient for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 56,'XPORT_COEFF_RMS',1,'FLOAT','Compensation Coefficient RMS',       'Compensation Coefficient RMS for Blue Laser runs'); 


/* LMF_LASER_IRED_COEFF_DAT       */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 57,'XPORT_COEFF',1,'FLOAT','Compensation Coefficient',       'Compensation Coefficient for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 57,'XPORT_COEFF_RMS',1,'FLOAT','Compensation Coefficient RMS',       'Compensation Coefficient RMS for Infra-Red Laser runs'); 


/* LMF_LASER_BLUE_SHAPE_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 58,'ALPHA',1,'FLOAT','Alpha',       'Alpha for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 58,'ALPHA_RMS',1,'FLOAT','Alpha RMS',       'Alpha RMS for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 58,'BETA',1,'FLOAT','Beta',       'Beta for Blue Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 58,'BETA_RMS',1,'FLOAT','Beta RMS',       'Beta RMS for Blue Laser runs'); 

/* LMF_LASER_IRED_SHAPE_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 59,'ALPHA',1,'FLOAT','Alpha',       'Alpha for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 59,'ALPHA_RMS',1,'FLOAT','Alpha RMS',       'Alpha RMS for Infra-Red Laser runs'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 59,'BETA',1,'FLOAT','Beta',       'Beta for Infra-Red Laser runs'); 

INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 59,'BETA_RMS',1,'FLOAT','Beta RMS',       'Beta RMS for Infra-Red Laser runs'); 

/* LMF_PN_BLUE_DAT    */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 60,'PN_PEAK',1,'FLOAT','Mean PN height',       'Mean PN Height in Blue Laser run (ADC Counts)'); 

INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 60,'PN_ERR',1,'FLOAT','Mean PN RMS',           'Mean PN RMS in Blue Laser run (ADC Counts)');

/* LMF_PN_IRED_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 61,'PN_PEAK',1,'FLOAT','Mean PN height',       'Mean PN Height in Infra-Red Laser run (ADC Counts)');
 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 61,'PN_ERR',1,'FLOAT','Mean PN RMS',           'Mean PN RMS in Infra-Red Laser run (ADC Counts)');

/* LMF_PN_TEST_PULSE_DAT       */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 62,'ADC_MEAN',1,'FLOAT','Mean PN height',       'Mean PN Height in Test-Pulse run (ADC Counts)'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 62,'ADC_RMS',1,'FLOAT','Mean PN RMS',           'Mean PN RMS in Test-Pulse run (ADC Counts)');

/* LMF_PN_CONFIG_DAT       */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 63, 'PNA_ID',1,'INT','PN 0 identity',       'PN used as PN 0 in the calculation'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 63, 'PNB_ID',1,'INT','PN 1 identity',       'PN used as PN 0 in the calculation'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 63, 'PNA_VALIDITY',1,'BOOLEAN','PN 0 validity',       'PN 0 validity'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 63, 'PNB_VALIDITY',1,'BOOLEAN','PN 1 validity',       'PN 1 validity'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 63, 'PNMEAN_VALIDITY',1,'BOOLEAN','PN Mean validity',       'PN Mean validity'); 
