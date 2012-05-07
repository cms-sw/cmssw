CREATE TABLE COND_TABLE_META (
DEF_ID NUMBER(10) NOT NULL,
TABLE_NAME VARCHAR2(30),
TABLE_TYPE NUMBER(10),
NUMBER_OF_FIELDS NUMBER(10),
FILLED_BY VARCHAR2(30),
CONTENT_EXPLANATION VARCHAR2(100), 
LOGIC_ID_EXPLANATION VARCHAR2(100),
LOGIC_ID_NAME VARCHAR2(50),
MAP_TO_BE_DONE_BY_i NUMBER(10),
MAP_TO_BE_DONE_BY_j NUMBER(10),
MAP_BY_LOGIC_ID_NAME VARCHAR2(50)
);

CREATE SEQUENCE COND_TABLE_SQ INCREMENT BY 1 START WITH 1;

ALTER TABLE COND_TABLE_META ADD CONSTRAINT COND_TABLE_META_PK PRIMARY KEY (DEF_ID);

CREATE TABLE COND_FIELD_META (
DEF_ID NUMBER(10) NOT NULL,
TAB_ID NUMBER(10) NOT NULL,
FIELD_NAME VARCHAR2(30),
IS_PLOTTABLE CHAR(1),
FIELD_TYPE VARCHAR2(15),
CONTENT_EXPLANATION VARCHAR2(100), 
LABEL VARCHAR2(80),
histo_min number,
histo_max number
); 

CREATE SEQUENCE COND_FIELD_SQ INCREMENT BY 1 START WITH 1;

ALTER TABLE COND_FIELD_META ADD CONSTRAINT COND_FIELD_META_PK PRIMARY KEY (DEF_ID);

ALTER TABLE COND_FIELD_META ADD CONSTRAINT COND_FIELD_to_COND_TABLE_fk FOREIGN KEY (TAB_ID) REFERENCES COND_TABLE_META (DEF_ID);

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

INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_TIMING_CRYSTAL_DAT',2,5,'DQM','Crystal Timing','Crystal numbering', 'EB_crystal_numbering',85,20);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_TIMING_TT_DAT',2,5,'DQM','TT Timing','TT numbering', 'EB_TT_numbering',17,4);                                                                                  


INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_LED1_DAT',2,5,'DQM','Endcap LED1 analysis','crystal numbering','EE_crystal_number',100,100);                                                                                  

INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_LED2_DAT',2,5,'DQM','Endcap LED2 analysis','crystal numbering','EE_crystal_number',100,100);                                                                                  

INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_LED1_DAT',2,9,'DQM','PN analysis in LED1 laser runs','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'MON_PN_LED2_DAT',2,9,'DQM','PN analysis in LED2 runs','PN numbering', 'EB_LM_PN',5,2);                                                                                  
INSERT INTO COND_TABLE_META(DEF_ID, TABLE_NAME, TABLE_TYPE,NUMBER_OF_FIELDS, FILLED_BY, CONTENT_EXPLANATION, LOGIC_ID_EXPLANATION, LOGIC_ID_NAME,MAP_TO_BE_DONE_BY_i, MAP_TO_BE_DONE_BY_j) values (COND_TABLE_SQ.NextVal, 'DCU_CCS_DAT',2,12,'CCS_Supervisor','Membox values and CCS temperatures','DCC numbering','ECAL_DCC',1,1);                                                                                  


/* RUN_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 9,'NUM_EVENTS',0,'NUMBER','Number of Events in a Run','Number of Events'); 

/* MON RUN_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'NUM_EVENTS',0,'NUMBER','Number of Events in a Run','Number of Events'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'RUN_OUTCOME_ID',0,'NUMBER','Global run quality','Global Run Quality'); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'ROOTFILE_NAME',0,'STRING','DQM plots root file','DQM plots root file');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'TASK_LIST',0,'INT','Active tasks in run','Active tasks in run (binary flags)');
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL) values (COND_FIELD_SQ.NextVal, 13,'TASK_OUTCOME',0,'INT','Task run quality','Task run quality (binary flags)');

/* MON_CRYSTAL_STATUS_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max ) values (COND_FIELD_SQ.NextVal,15, 'STATUS_G1',1,'INT','Channel Status Gain1','Channel Status Gain1',0,1);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max ) values (COND_FIELD_SQ.NextVal,15, 'STATUS_G6',1,'INT','Channel Status Gain6','Channel Status Gain6',0,1);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max ) values (COND_FIELD_SQ.NextVal,15, 'STATUS_G12',1,'INT','Channel Status Gain12','Channel Status Gain12',0,1);

/* MON_PN_STATUS_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max ) values (COND_FIELD_SQ.NextVal, 17 , 'STATUS_G1',1,'INT','Channel Status Gain1','Channel Status Gain1',0,1);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max ) values (COND_FIELD_SQ.NextVal, 17 , 'STATUS_G16',1,'INT','Channel Status Gain16','Channel Status Gain16',0,1);

/* MON_CRYSTAL_CONSISTENCY_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 18, 'PROCESSED_EVENTS',1,'INT','Number of processed events','Number of processed events',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 18, 'PROBLEMATIC_EVENTS',1,'INT','Number of problematic events','Number of problematic events',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 18, 'PROBLEMS_ID',1,'INT','Channel wrong identification','Channel wrong identification',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 18, 'PROBLEMS_GAIN_ZERO',1,'INT','Channel gain==0','Channel gain==0',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 18, 'PROBLEMS_GAIN_SWITCH',1,'INT','Channel forbidden gain switch','Channel forbidden gain switch',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 18, 'TASK_STATUS',1,'INT','Integrity Channel status','Integrity Channel status',0,0);

/* MON_TT_CONSISTENCY_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 19, 'PROCESSED_EVENTS',1,'INT','Number of processed events','Number of processed events',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMATIC_EVENTS',1,'INT','Number of problematic events','Number of problematic events',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMS_ID',1,'INT','Trigger tower wrong identification','Trigger tower wrong identification',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMS_SIZE',1,'INT','Trigger Tower wrong size','Trigger Tower wrong size',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMS_LV1',1,'INT','LV1 wrong in Trigger Tower','LV1 wrong in Trigger Tower',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 19, 'PROBLEMS_BUNCH_X',1,'INT','BunchX wrong in Trigger Tower','BunchX wrong in Trigger Tower',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 19, 'TASK_STATUS',1,'INT','Integrity Tower status','Integrity Tower status',0,0);

/* MON_OCCUPANCY_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 20 ,'EVENTS_OVER_LOW_THRESHOLD',1,'INT','Number of events above low energy threshold','Number of events above low threshold',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 20 ,'EVENTS_OVER_HIGH_THRESHOLD',1,'INT','Number of events above high energy threshold','Number of events above high threshold',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 20 ,'AVG_ENERGY',1,'FLOAT','Average energy per channel','Average energy per channel (ADC Counts)',0.0,150.0);


/* MON_PEDESTALS_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 21,'PED_MEAN_G1',1,'FLOAT','Pedestal gain 1','Pedestal Gain1 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 21,'PED_MEAN_G6',1,'FLOAT','Pedestal gain 6','Pedestal Gain6 (ADC Counts)',180.,220);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 21,'PED_MEAN_G12',1,'FLOAT','Pedestal gain 12','Pedestal Gain12 (ADC Counts)',180.,220);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 21,'PED_RMS_G1',1,'FLOAT','Pedestal RMS gain 1','RMS Pedestal Gain1 (ADC Counts)',0.4, 0.8);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 21,'PED_RMS_G6',1,'FLOAT','Pedestal RMS gain 6','RMS Pedestal Gain6 (ADC Counts)', 0.5, 0.9);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 21,'PED_RMS_G12',1,'FLOAT','Pedestal RMS gain 12','RMS Pedestal Gain12 (ADC Counts)', 0.8,1.8 );
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 21, 'TASK_STATUS',1,'INT','Pedestal Channel status','Pedestal Channel status',0,1);

/* MON_PEDESTALS_ONLINE_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 22,'ADC_MEAN_G12',1,'FLOAT','Online Pedestal gain 12','Online Pedestal Gain12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 22,'ADC_RMS_G12',1,'FLOAT','Online Pedestal RMS gain 12','RMS Online Pedestal Gain12 (ADC Counts)',0.5,2.5);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 22,'TASK_STATUS',1,'INT','Online Pedestal Channel status','Online Pedestal Channel status',0,1);


/* MON_PEDESTAL_OFFSETS_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,23 ,'DAC_G1',1,'INT','DAC for Pedestal gain 1','DAC for Pedestal Gain1',0,300);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 23,'DAC_G6',1,'INT','DAC for Pedestal gain 6','DAC for Pedestal Gain6',0,300);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 23,'DAC_G12',1,'INT','DAC for Pedestal gain 12','DAC for Pedestal Gain12',0,300); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 23, 'TASK_STATUS',1,'INT','Task status','Task status',0,1);


/* MON_TEST_PULSE_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 24,'ADC_MEAN_G1',1,'FLOAT','Mean Test-pulse Gain 1','Mean Test-pulse Gain 1 (ADC Counts)',19200.,20800.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 24,'ADC_RMS_G1',1,'FLOAT','RMS Test-pulse Gain 1','RMS Test-pulse Gain 1(ADC Counts)',0.,25.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 24,'ADC_MEAN_G6',1,'FLOAT','Mean Test-pulse Gain 6','Mean Test-pulse Gain 6 (ADC Counts)',5400.,5800.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 24,'ADC_RMS_G6',1,'FLOAT','RMS Test-pulse Gain 6','RMS Test-pulse Gain 6 (ADC Counts)',0.,6.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 24,'ADC_MEAN_G12',1,'FLOAT','Mean Test-pulse Gain 12','Mean Test-pulse Gain 12 (ADC Counts)',1900.,2100.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 24,'ADC_RMS_G12',1,'FLOAT','RMS Test-pulse Gain 12','RMS Test-pulse Gain 12 (ADC Counts)',0.,3.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 24, 'TASK_STATUS',1,'INT','Test-pulse Channel status','Test-pulse Channel status',0,1);


 
/*  MON_PULSE_SHAPE_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_01',1,'FLOAT','Mean height sample 01 Gain 1','Mean height sample 01 Gain 1 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_02',1,'FLOAT','Mean height sample 02 Gain 1','Mean height sample 02 Gain 1 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_03',1,'FLOAT','Mean height sample 03 Gain 1','Mean height sample 03 Gain 1 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_04',1,'FLOAT','Mean height sample 04 Gain 1','Mean height sample 04 Gain 1 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_05',1,'FLOAT','Mean height sample 05 Gain 1','Mean height sample 05 Gain 1 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_06',1,'FLOAT','Mean height sample 06 Gain 1','Mean height sample 06 Gain 1 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_07',1,'FLOAT','Mean height sample 07 Gain 1','Mean height sample 07 Gain 1 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_08',1,'FLOAT','Mean height sample 08 Gain 1','Mean height sample 08 Gain 1 (ADC Counts)',180.,220);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_09',1,'FLOAT','Mean height sample 09 Gain 1','Mean height sample 09 Gain 1 (ADC Counts)',180.,220);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G1_AVG_SAMPLE_10',1,'FLOAT','Mean height sample 10 Gain 1','Mean height sample 10 Gain 1 (ADC Counts)',180.,220.);
/* samples at gain 6 */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_01',1,'FLOAT','Mean height sample 01 Gain 6','Mean height sample 01 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_02',1,'FLOAT','Mean height sample 02 Gain 6','Mean height sample 02 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_03',1,'FLOAT','Mean height sample 03 Gain 6','Mean height sample 03 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_04',1,'FLOAT','Mean height sample 04 Gain 6','Mean height sample 04 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_05',1,'FLOAT','Mean height sample 05 Gain 6','Mean height sample 05 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_06',1,'FLOAT','Mean height sample 06 Gain 6','Mean height sample 06 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_07',1,'FLOAT','Mean height sample 07 Gain 6','Mean height sample 07 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_08',1,'FLOAT','Mean height sample 08 Gain 6','Mean height sample 08 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_09',1,'FLOAT','Mean height sample 09 Gain 6','Mean height sample 09 Gain 6 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G6_AVG_SAMPLE_10',1,'FLOAT','Mean height sample 10 Gain 6','Mean height sample 10 Gain 6 (ADC Counts)',180.,220.);
/* samples at gain 12 */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_01',1,'FLOAT','Mean height sample 01 Gain 12','Mean height sample 01 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_02',1,'FLOAT','Mean height sample 02 Gain 12','Mean height sample 02 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_03',1,'FLOAT','Mean height sample 03 Gain 12','Mean height sample 03 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_04',1,'FLOAT','Mean height sample 04 Gain 12','Mean height sample 04 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_05',1,'FLOAT','Mean height sample 05 Gain 12','Mean height sample 05 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_06',1,'FLOAT','Mean height sample 06 Gain 12','Mean height sample 06 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_07',1,'FLOAT','Mean height sample 07 Gain 12','Mean height sample 07 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_08',1,'FLOAT','Mean height sample 08 Gain 12','Mean height sample 08 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_09',1,'FLOAT','Mean height sample 09 Gain 12','Mean height sample 09 Gain 12 (ADC Counts)',180.,220.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 25,'G12_AVG_SAMPLE_10',1,'FLOAT','Mean height sample 10 Gain 12','Mean height sample 10 Gain 12 (ADC Counts)',180.,220.);


/* MON_SHAPE_QUALITY_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 26,'AVG_CHI2',1,'FLOAT','Average chi2 of the pulse fit','Fit Average Chi2',0,0);

/* MON_DELAYS_TT_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 27,'DELAY_MEAN',1,'FLOAT','Mean Delay per Trigger Tower','Mean TT Delay (ns)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 27,'DELAY_RMS',1,'FLOAT','RMS Delay per Trigger Tower','RMS TT Delay (ns)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 27, 'TASK_STATUS',1,'INT','Delays Tower status','Delays Tower status',0,1);

/* MON_PN_BLUE_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Blue Laser PN Amplitude Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Blue Laser PN RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Blue Laser PN Amplitude Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Blue Laser PN RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Blue Laser PN Pedestal Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Blue Laser PN Pedestal RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Blue Laser PN Pedestal Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Blue Laser PN Pedestal RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 28, 'TASK_STATUS',1,'INT','PN Blue status','PN Blue status',0,1);

/* MON_PN_GREEN_DAT */ 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Green Laser PN Amplitude Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Green Laser PN RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Green Laser PN Amplitude Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Green Laser PN RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Green Laser PN Pedestal Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Green Laser PN Pedestal RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Green Laser PN Pedestal Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Green Laser PN Pedestal RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 29, 'TASK_STATUS',1,'INT','PN Green status','PN Green status',0,1);


/* MON_PN_RED_DAT    */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Red Laser PN Amplitude Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Red Laser PN RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Red Laser PN Amplitude Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Red Laser PN RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Red Laser PN Pedestal Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Red Laser PN Pedestal RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Red Laser PN Pedestal Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Red Laser PN Pedestal RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 30, 'TASK_STATUS',1,'INT','PN Red status','PN Red status',0,1);



/* MON_PN_IRED_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Infra-Red Laser PN Amplitude Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Infra-Red Laser PN RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Infra-Red Laser PN Amplitude Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Infra-Red Laser PN RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Infra-Red Laser PN Pedestal Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Infra-Red Laser PN Pedestal RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Infra-Red Laser PN Pedestal Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Infra-Red Laser PN Pedestal RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 31,'TASK_STATUS',1,'INT','PN Infra-Red status','PN Infra-Red status',0,1);




/* MON_PN_PED_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,32,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',   'Mean PN Pedestal Gain 1 (ADC Counts)',700.,800.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,32,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1','Mean PN Pedestal RMS Gain 1 (ADC Counts)',0.4,0.6);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,32,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean PN Pedestal Gain 16 (ADC Counts)',700.,800.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,32,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean PN Pedestal RMS Gain 16 (ADC Counts)',1.4,2.0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,32,'TASK_STATUS',1,'INT','PN Pedestal status','PN Pedestal status',0,1);

/* MON_PN_MGPA_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Test-Pulse PN Amplitude Gain 1 (ADC Counts)',0.5,1.5);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Test-Pulse PN RMS Gain 1 (ADC Counts)',0.3,0.6);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Test-Pulse PN Amplitude Gain 16 (ADC Counts)',1.5,3.0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Test-Pulse PN RMS Gain 16 (ADC Counts)',0.6,0.8);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean PN Pedestal from Test-Pulse runs Gain 1 (ADC Counts)',700.,800.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean PN Pedestal from Test-Pulse runs RMS Gain 1 (ADC Counts)',0.5,1.5);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean PN Pedestal from Test-Pulse runs Gain 16 (ADC Counts)',700.,800.);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean PN Pedestal from Test-Pulse runs RMS Gain 16 (ADC Counts)',1.5,3.0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 33,'TASK_STATUS',1,'INT','PN Test-Pulse status','PN Test-Pulse status',0,1);


/* MON_LASER_BLUE_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 34,'APD_MEAN',1,'FLOAT','Mean height',       'Mean Channel Amplitude for Blue Laser runs (ADC Counts)',0.,0.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 34,'APD_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Blue Laser runs (ADC Counts)',40.,120.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 34,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PN',       'Mean APD/PN for Blue Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 34,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PN',       'RMS APD/PN for Blue Laser runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 34,'TASK_STATUS',1,'INT','Blue Laser Channel status','Blue Laser Channel status',0,1);

/* MON_LASER_GREEN_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 35,'APD_MEAN',1,'FLOAT','Mean height',       'Mean Channel Amplitude for Green Laser runs (ADC Counts)',1500.,4000.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 35,'APD_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Green Laser runs (ADC Counts)',40.,120.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 35,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PN',       'Mean APD/PN for Green Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 35,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PN',       'RMS APD/PN for Green Laser runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 35,'TASK_STATUS',1,'INT','Green Laser Channel status','Green Laser Channel status',0,1);

/* MON_LASER_RED_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 36,'APD_MEAN',1,'FLOAT','Mean height',       'Mean Channel Amplitude for Red Laser runs (ADC Counts)',1500.,4000.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 36,'APD_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Red Laser runs (ADC Counts)',40.,120.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 36,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PN',       'Mean APD/PN for Red Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 36,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PN',       'RMS APD/PN for Red Laser runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 36,'TASK_STATUS',1,'INT','Red Laser Channel status','Red Laser Channel status',0,1);

/* MON_LASER_IRED_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 37,'APD_MEAN',1,'FLOAT','Mean height',       'Mean Channel Amplitude for Infra-Red Laser runs (ADC Counts)',1500.,4000.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 37,'APD_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Infra-Red Laser runs (ADC Counts)',40.,120.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 37,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PN',       'Mean APD/PN for Infra-Red Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 37,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PN',       'RMS APD/PN for Infra-Red Laser runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 37,'TASK_STATUS',1,'INT','Infra-Red Laser Channel status','Infra-Red Laser Channel status',0,1);


/* MON_MEM_CH_CONSISTENCY_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 38, 'PROCESSED_EVENTS',1,'INT','Number of processed events','Number of processed events',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 38, 'PROBLEMATIC_EVENTS',1,'INT','Number of problematic events','Number of problematic events',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 38, 'PROBLEMS_ID',1,'INT','MEM Channel wrong identification','Mem Channel wrong identification',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 38, 'PROBLEMS_GAIN_ZERO',1,'INT','MEM Channel gain==0','MEM Channel gain==0',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 38, 'PROBLEMS_GAIN_SWITCH',1,'INT','MEM Channel forbidden gain switch','MEM Channel forbidden gain switch',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 38, 'TASK_STATUS',1,'INT','Task status','Task status',0,1);


/* MON_MEM_TT_CONSISTENCY_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 39, 'PROCESSED_EVENTS',1,'INT','Number of processed events','Number of processed events',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMATIC_EVENTS',1,'INT','Number of problematic events','Number of problematic events',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMS_ID',1,'INT','MEM TTBlock wrong identification','MEM TTBlock wrong identification',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMS_SIZE',1,'INT','MEM TTBlock wrong size','MEM TTBlock wrong size',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMS_LV1',1,'INT','MEM TTBlock wrong LV1','MEM TTBlock wrong LV1',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 39, 'PROBLEMS_BUNCH_X',1,'INT','MEM TTBlock wrong BunchX','MEM TTBlock wrong BunchX',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 39, 'TASK_STATUS',1,'INT','MEM TTBlock Integrity status','MEM TTBlock Integrity status',0,1);

/* DCU_CCS_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M1_VDD1',1,'FLOAT','FE Board Voltage 1 for Mem 1','Mem 1 VDD1 (V)',1.,3.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M2_VDD1',1,'FLOAT','FE Board Voltage 1 for Mem 2','Mem 2 VDD1 (V)',1.,3.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M1_VDD2',1,'FLOAT','FE Board Voltage 2 for Mem 1','Mem 1 VDD2 (V)',1.,3.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M2_VDD2',1,'FLOAT','FE Board Voltage 2 for Mem 2','Mem 2 VDD2 (V)',1.,3.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M1_VINJ',1,'FLOAT','FE Board Test Pulse Voltage for Mem 1','Mem 1 Vinj (V)',0.,1.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M2_VINJ',1,'FLOAT','FE Board Test Pulse Voltage for Mem 2','Mem 2 Vinj (V)',0.,1.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M1_VCC',1,'FLOAT','ADC Voltage for Mem 1','Mem 1 ADC Vcc (V)',3.,6.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M2_VCC',1,'FLOAT','ADC Voltage for Mem 2','Mem 2 ADC Vcc (V)',3.,6.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M1_DCUTEMP',1,'FLOAT','DCU Chip Temperature','Chip Temperature 1 (C)',30.,50.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'M2_DCUTEMP',1,'FLOAT','DCU Chip Temperature','Chip Temperature 2 (C)',30.,50.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'CCSTEMPLOW',1,'FLOAT','Temperature of the low sensor in the CCS board','CCS Low Sensor Temp. (C)',18.,35.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,70, 'CCSTEMPHIGH',1,'FLOAT','Temperature of the high sensor in the CCS board','CCS High Sensor Temp. (C)',18.,35.); 

/* DCU_CAPSULE_TEMP_DAT     */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,42, 'CAPSULE_TEMP',1,'FLOAT','Capsule Temperature',       'Capsule Temperature (C)',17.,21.); 

/* DCU_CAPSULE_TEMP_RAW_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,43, 'CAPSULE_TEMP_ADC',1,'FLOAT','Raw Capsule Temperature',       'Raw Capsule Temperature (ADC Counts)',2900.,3300.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,43, 'CAPSULE_TEMP_RMS',1,'FLOAT','Raw Capsule Temperature RMS',       'Raw Capsule Temperature RMS (ADC Counts)',0,0); 

/* DCU_IDARK_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,44, 'APD_IDARK',1,'FLOAT','Capsule Dark Current',       'Capsule Dark Current (microAmpere)',0.,10.); 

/* DCU_IDARK_PED_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,45,'PED',1,'FLOAT','Capsule Dark Current DCU Pedestal',       'Capsule Dark Current DCU Pedestal (ADC Counts)',0,0); 

/* DCU_VFE_TEMP_DAT   */ 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,46,'VFE_TEMP',1,'FLOAT','Temperature of DCU chip on VFEs',       'Temperature (C) of DCU chip on VFEs',30.,40.);   

/* DCU_LVR_TEMPS_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,47 ,'T1',1,'FLOAT','Temperature of DCU chip 1 on LVR',       'Temperature (C) of DCU chip 1 on LVR',35.,50.);  

INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,47 ,'T2',1,'FLOAT','Temperature (C) of DCU chip 2 on LVR',       'Temperature (C) of DCU chip 2 on LVR',35.,50.); 
 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,47 ,'T3',1,'FLOAT','Temperature (C) of DCU chip 3 on LVR',       'Temperature (C) of DCU chip 3 on LVR',35.,50.);  
 
/* DCU_LVRB_TEMPS_DAT     */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,48 ,'T1',1,'FLOAT','LVRB Temperatures',       'LVRB Temperature Sensor 1 (C)',20.,40.);  
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,48 ,'T2',1,'FLOAT','LVRB Temperatures',       'LVRB Temperature Sensor 2 (C)',20.,40.);  
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,48 ,'T3',1,'FLOAT','LVRB Temperatures',       'LVRB Temperature Sensor 3 (C)',20.,40.);  


/* DCU_LVR_VOLTAGES_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'VFE1_A',1,'FLOAT','VFE 1 Analog Voltage',       'VFE 1 Analog Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'VFE2_A',1,'FLOAT','VFE 2 Analog Voltage',       'VFE 2 Analog Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'VFE3_A',1,'FLOAT','VFE 3 Analog Voltage',       'VFE 3 Analog Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'VFE4_A',1,'FLOAT','VFE 4 Analog Voltage',       'VFE 4 Analog Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'VFE5_A',1,'FLOAT','VFE 5 Analog Voltage',       'VFE 5 Analog Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'VCC',1,'FLOAT','VCC',       'VCC (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'VFE4_5_D',1,'FLOAT','VFE 4 and 5 Digital Voltage',       'VFE 4 and 5 Digital Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'VFE1_2_3_D',1,'FLOAT','VFE 1 2 and 3 Digital Voltage',       'VFE 1 2 and 3 Digital Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'BUFFER',1,'FLOAT','Buffer Voltage',       'Buffer Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'FENIX',1,'FLOAT','Fenix Voltage',       'Fenix Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'V43_A',1,'FLOAT','Input 4.3 V Analog Voltage',       'Input 4.3 V Analog Voltage (V)',2.3,6.3);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'OCM',1,'FLOAT','OCM',       'OCM Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'GOH',1,'FLOAT','GOH',       'GOH Voltage (V)',2.3,2.7);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'INH',1,'FLOAT','Inhibit',       'Inhibit Voltage (V)',0.,5.);   
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal,49,'V43_D',1,'FLOAT','Input 4.3 V Digital Voltage',       'Input 4.3 V Digital Voltage (V)',2.3,6.3);   


/* LMF_LASER_BLUE_RAW_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 52,'APD_PEAK',1,'FLOAT','Mean height',       'Mean Channel Amplitude for Blue Laser runs (ADC Counts)',1500.,4000.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 52,'APD_ERR',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Blue Laser runs (ADC Counts)',40.,120.); 

/* LMF_LASER_IRED_RAW_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 53,'APD_PEAK',1,'FLOAT','Mean height',       'Mean Channel Amplitude for Infra-Red Laser runs (ADC Counts)',1500.,4000.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 53,'APD_ERR',1,'FLOAT','Mean RMS',       'Mean Channel RMS for Infra-Red Laser runs (ADC Counts)',40.,120.); 

/* LMF_LASER_BLUE_NORM_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PNA_MEAN',1,'FLOAT','Mean APD/PN 0',       'Mean APD/PN0 for Blue Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PNA_RMS',1,'FLOAT','RMS APD/PN 0',       'RMS APD/PN0 for Blue Laser runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PNB_MEAN',1,'FLOAT','Mean APD/PN 1',       'Mean APD/PN1 for Blue Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PNB_RMS',1,'FLOAT','RMS APD/PN 1',       'RMS APD/PN1 for Blue Laser runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PNmean',       'Mean APD/PNmean for Blue Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 54,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PNmean',       'RMS APD/PNmean for Blue Laser runs',0.,0.2); 



/* LMF_LASER_IRED_NORM_DAT    */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PNA_MEAN',1,'FLOAT','Mean APD/PN 0',       'Mean APD/PN0 for Infra-Red Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PNA_RMS',1,'FLOAT','RMS APD/PN 0',       'RMS APD/PN0 for Infra-Red Laser runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PNB_MEAN',1,'FLOAT','Mean APD/PN 1',       'Mean APD/PN1 for Infra-Red Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PNB_RMS',1,'FLOAT','RMS APD/PN 1',       'RMS APD/PN1 for Infra-Red Laser runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PN_MEAN',1,'FLOAT','Mean APD/PNmean',       'Mean APD/PNmean for Infra-Red Laser runs',6.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 55,'APD_OVER_PN_RMS',1,'FLOAT','RMS APD/PNmean',       'RMS APD/PNmean for Infra-Red Laser runs',0.,0.2); 


/* LMF_LASER_BLUE_COEFF_DAT    */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 56,'XPORT_COEFF',1,'FLOAT','Compensation Coefficient',       'Compensation Coefficient for Blue Laser runs',0,2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 56,'XPORT_COEFF_RMS',1,'FLOAT','Compensation Coefficient RMS',       'Compensation Coefficient RMS for Blue Laser runs',0,0); 


/* LMF_LASER_IRED_COEFF_DAT       */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 57,'XPORT_COEFF',1,'FLOAT','Compensation Coefficient',       'Compensation Coefficient for Infra-Red Laser runs',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 57,'XPORT_COEFF_RMS',1,'FLOAT','Compensation Coefficient RMS',       'Compensation Coefficient RMS for Infra-Red Laser runs',0,0); 


/* LMF_LASER_BLUE_SHAPE_DAT   */	
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 58,'ALPHA',1,'FLOAT','Alpha',       'Alpha for Blue Laser runs',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 58,'ALPHA_RMS',1,'FLOAT','Alpha RMS',       'Alpha RMS for Blue Laser runs',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 58,'BETA',1,'FLOAT','Beta',       'Beta for Blue Laser runs',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 58,'BETA_RMS',1,'FLOAT','Beta RMS',       'Beta RMS for Blue Laser runs',0,0); 

/* LMF_LASER_IRED_SHAPE_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 59,'ALPHA',1,'FLOAT','Alpha',       'Alpha for Infra-Red Laser runs',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 59,'ALPHA_RMS',1,'FLOAT','Alpha RMS',       'Alpha RMS for Infra-Red Laser runs',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 59,'BETA',1,'FLOAT','Beta',       'Beta for Infra-Red Laser runs',0,0); 

INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 59,'BETA_RMS',1,'FLOAT','Beta RMS',       'Beta RMS for Infra-Red Laser runs',0,0); 

/* LMF_PN_BLUE_DAT    */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 60,'PN_PEAK',1,'FLOAT','Mean PN height',       'Mean PN Amplitude in Blue Laser run (ADC Counts)',0,0); 

INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 60,'PN_ERR',1,'FLOAT','Mean PN RMS',           'Mean PN RMS in Blue Laser run (ADC Counts)',0,0);

/* LMF_PN_IRED_DAT   */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 61,'PN_PEAK',1,'FLOAT','Mean PN height',       'Mean PN Amplitude in Infra-Red Laser run (ADC Counts)',0,0);
 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 61,'PN_ERR',1,'FLOAT','Mean PN RMS',           'Mean PN RMS in Infra-Red Laser run (ADC Counts)',0,0);

/* LMF_PN_TEST_PULSE_DAT       */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 62,'ADC_MEAN',1,'FLOAT','Mean PN height',       'Mean PN Amplitude in Test-Pulse run (ADC Counts)',0.5,1.5); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 62,'ADC_RMS',1,'FLOAT','Mean PN RMS',           'Mean PN RMS in Test-Pulse run (ADC Counts)',0.3,0.5);

/* LMF_PN_CONFIG_DAT       */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 63, 'PNA_ID',1,'INT','PN 0 identity',       'PN used as PN 0 in the calculation',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 63, 'PNB_ID',1,'INT','PN 1 identity',       'PN used as PN 0 in the calculation',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 63, 'PNA_VALIDITY',1,'BOOLEAN','PN 0 validity',       'PN 0 validity',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 63, 'PNB_VALIDITY',1,'BOOLEAN','PN 1 validity',       'PN 1 validity',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 63, 'PNMEAN_VALIDITY',1,'BOOLEAN','PN Mean validity',       'PN Mean validity',0,0); 

/* MON_TIMING_CRYSTAL_DAT       */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 64, 'TIMING_MEAN',1,'FLOAT','Mean Crystal Timing',       'Mean Crystal Timing (ns)',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 64, 'TIMING_RMS',1,'FLOAT','RMS Crystal Timing',       'RMS Crystal Timing (ns)',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 64, 'TASK_STATUS',1,'INT','Crystal Timing Task Status', 'Crystal Timing Task Status',0,1);
/* MON_TIMING_TT_DAT       */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 65, 'TIMING_MEAN',1,'FLOAT','Mean TT Timing',       'Mean TT Timing (ns)',0,0); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 65, 'TIMING_RMS',1,'FLOAT','RMS TT Timing',       'RMS TT Timing (ns)',0,0); 

INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 65, 'TASK_STATUS',1,'INT','TT Timing Task Status', 'TT Timing Task Status',0,1);

/* MON_LED1_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 66,'VPT_MEAN',1,'FLOAT','Mean height',       'Mean Channel Amplitude for LED1 runs (ADC Counts)',0.,0.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 66,'VPT_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for LED1 runs (ADC Counts)',40.,120.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 66,'VPT_OVER_PN_MEAN',1,'FLOAT','Mean VPT/PN',       'Mean VPT/PN for LED1 runs',0.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 66,'VPT_OVER_PN_RMS',1,'FLOAT','RMS VPT/PN',       'RMS VPT/PN for LED1 runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 66,'TASK_STATUS',1,'INT','LED1 Channel status','LED1 Channel status',0,1);
/* MON_LED2_DAT  */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 67,'VPT_MEAN',1,'FLOAT','Mean height',       'Mean Channel Amplitude for LED2 runs (ADC Counts)',0.,0.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 67,'VPT_RMS',1,'FLOAT','Mean RMS',       'Mean Channel RMS for LED2 runs (ADC Counts)',40.,120.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 67,'VPT_OVER_PN_MEAN',1,'FLOAT','Mean VPT/PN',       'Mean VPT/PN for LED2 runs',0.,10.); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 67,'VPT_OVER_PN_RMS',1,'FLOAT','RMS VPT/PN',       'RMS VPT/PN for LED2 runs',0.,0.2); 
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 67,'TASK_STATUS',1,'INT','LED2 Channel status','LED1 Channel status',0,1);


/* MON_PN_LED1_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Led1 PN Amplitude Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Led1 PN RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Led1 PN Amplitude Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Led1 PN RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Led1 PN Pedestal Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Led1 PN Pedestal RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Led1 PN Pedestal Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Led1 PN Pedestal RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 68, 'TASK_STATUS',1,'INT','PN Led1 status','PN Led1 status',0,1);

/* MON_PN_LED2_DAT */
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69,'ADC_MEAN_G1',1,'FLOAT','Mean PN height Gain1',       'Mean Led2 PN Amplitude Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69,'ADC_RMS_G1',1,'FLOAT','Mean PN RMS Gain1',           'Mean Led2 PN RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69,'ADC_MEAN_G16',1,'FLOAT','Mean PN height Gain16',     'Mean Led2 PN Amplitude Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69,'ADC_RMS_G16',1,'FLOAT','Mean PN RMS Gain16',         'Mean Led2 PN RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69,'PED_MEAN_G1',1,'FLOAT','Mean PN Pedestal Gain1',     'Mean Led2 PN Pedestal Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69,'PED_RMS_G1',1,'FLOAT','Mean PN Pedestal RMS Gain1',  'Mean Led2 PN Pedestal RMS Gain 1 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69,'PED_MEAN_G16',1,'FLOAT','Mean PN Pedestal Gain16',   'Mean Led2 PN Pedestal Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69,'PED_RMS_G16',1,'FLOAT','Mean PN Pedestal RMS Gain16','Mean Led2 PN Pedestal RMS Gain 16 (ADC Counts)',0,0);
INSERT INTO COND_FIELD_META(DEF_ID, TAB_ID, FIELD_NAME, IS_PLOTTABLE, FIELD_TYPE, CONTENT_EXPLANATION, LABEL, histo_min, histo_max) values (COND_FIELD_SQ.NextVal, 69, 'TASK_STATUS',1,'INT','PN Led2 status','PN Led2 status',0,1);
