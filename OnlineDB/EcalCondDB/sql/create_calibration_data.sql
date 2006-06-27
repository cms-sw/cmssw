/*
 *  Creates all the data tables referencing calibration_iov
 *  Requires:  create_calibration_core.sql
 */



CREATE TABLE cali_general_dat (
  iov_id                NUMBER(10),
  logic_id          	NUMBER(10), -- (SM)
  NUM_EVENTS          	NUMBER(10), 
  location_id		NUMBER(10) NOT NULL,
  comment               VARCHAR(100)  
);
 
ALTER TABLE cali_general_dat ADD CONSTRAINT cali_general_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE cali_general_dat ADD CONSTRAINT cali_general_fk FOREIGN KEY (iov_id) REFERENCES cali_iov (iov_id);
ALTER TABLE cali_general_tag ADD CONSTRAINT cali_general_fk1 FOREIGN KEY (location_id) REFERENCES location_def (def_id);


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
