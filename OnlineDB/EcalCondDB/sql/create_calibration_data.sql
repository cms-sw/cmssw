/*
 *  Creates all the data tables referencing calibration_iov
 *  Requires:  create_calibration_core.sql
 */



CREATE TABLE cali_general_dat (
  iov_id                NUMBER(10),
  logic_id          	NUMBER(10), -- (SM)
  NUM_EVENTS          	NUMBER(10), 
  comments              VARCHAR(100)  
);
 
ALTER TABLE cali_general_dat ADD CONSTRAINT cali_general_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE cali_general_dat ADD CONSTRAINT cali_general_fk FOREIGN KEY (iov_id) REFERENCES cali_iov (iov_id);



CREATE TABLE cali_crystal_intercal_dat (
  iov_id                NUMBER(10),
  logic_id          	NUMBER(10), -- (crystal)
  cali	                BINARY_FLOAT,
  cali_rms       	BINARY_FLOAT,
  NUM_EVENTS          	NUMBER(10), 
  task_status           char(1) 	
);
 
ALTER TABLE cali_crystal_intercal_dat ADD CONSTRAINT cali_crystal_intercal_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE cali_crystal_intercal_dat ADD CONSTRAINT cali_crystal_intercal_fk FOREIGN KEY (iov_id) REFERENCES cali_iov (iov_id);

CREATE TABLE cali_hv_scan_ratio_dat (
  iov_id                NUMBER(10),
  logic_id          	NUMBER(10), -- (crystal)
  hvratio               BINARY_FLOAT,
  hvratio_rms       	BINARY_FLOAT,
  task_status           char(1) 	
);
 
ALTER TABLE cali_hv_scan_ratio_dat ADD CONSTRAINT cali_hv_scan_ratio_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE cali_hv_scan_ratio_dat ADD CONSTRAINT cali_hv_scan_ratio_fk FOREIGN KEY (iov_id) REFERENCES cali_iov (iov_id);

CREATE TABLE CALI_GAIN_RATIO_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10), -- (crystal)
  g1_g12                BINARY_FLOAT,
  g6_g12                BINARY_FLOAT,
  task_status           char(1)
);

ALTER TABLE CALI_GAIN_RATIO_DAT ADD CONSTRAINT CALI_GAIN_RATIO_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE CALI_GaIN_RATIO_DAT ADD CONSTRAINT CALI_GAIN_RATIO_fk FOREIGN KEY (iov_id) REFERENCES cali_iov (iov_id);

CREATE TABLE CALI_TEMP_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10), -- (crystal)
  beta                  BINARY_FLOAT,
  r25                   BINARY_FLOAT,
  offset                BINARY_FLOAT,
  task_status           char(1)
);

ALTER TABLE CALI_TEMP_DAT ADD CONSTRAINT CALI_TEMP_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE CALI_TEMP_DAT ADD CONSTRAINT CALI_TEMP_fk FOREIGN KEY (iov_id) REFERENCES cali_iov (iov_id);
