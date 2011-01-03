/*
 *  Creates all the data tables referencing dcu_iov
 *  Requires:  create_dcu_core.sql
 */




CREATE TABLE DCU_CAPSULE_TEMP_DAT (
  iov_id                NUMBER(10),
  logic_id          	NUMBER(10), -- capsule (crystal)
  capsule_temp		BINARY_FLOAT
);
 
ALTER TABLE DCU_CAPSULE_TEMP_DAT ADD CONSTRAINT DCU_CAPSULE_TEMP_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE DCU_CAPSULE_TEMP_DAT ADD CONSTRAINT DCU_CAPSULE_TEMP_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);



CREATE TABLE DCU_CAPSULE_TEMP_RAW_DAT (
  iov_id                NUMBER(10),
  logic_id          	NUMBER(10), -- capsule (crystal)
  capsule_temp_adc	BINARY_FLOAT,
  capsule_temp_rms	BINARY_FLOAT
);
 
ALTER TABLE DCU_CAPSULE_TEMP_RAW_DAT ADD CONSTRAINT DCU_CAPSULE_TEMP_RAW_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE DCU_CAPSULE_TEMP_RAW_DAT ADD CONSTRAINT DCU_CAPSULE_TEMP_RAW_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);


 
CREATE TABLE DCU_IDARK_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10),  -- crystal
  apd_idark          	BINARY_FLOAT
);
 
ALTER TABLE DCU_IDARK_DAT ADD CONSTRAINT DCU_IDARK_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE DCU_IDARK_DAT ADD CONSTRAINT DCU_IDARK_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);



CREATE TABLE DCU_IDARK_PED_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10),  -- crystal
  ped	          	BINARY_FLOAT
);
 
ALTER TABLE DCU_IDARK_PED_DAT ADD CONSTRAINT DCU_IDARK_PED_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE DCU_IDARK_PED_DAT ADD CONSTRAINT DCU_IDARK_PED_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);



CREATE TABLE DCU_VFE_TEMP_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10),  -- VFE card
  vfe_temp           	BINARY_FLOAT
);
 
ALTER TABLE DCU_VFE_TEMP_DAT ADD CONSTRAINT DCU_VFE_TEMP_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE DCU_VFE_TEMP_DAT ADD CONSTRAINT DCU_VFE_TEMP_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);



CREATE TABLE DCU_LVR_TEMPS_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10), -- TT
  t1           		BINARY_FLOAT,
  t2           		BINARY_FLOAT,
  t3           		BINARY_FLOAT
);
 
ALTER TABLE  DCU_LVR_TEMPS_DAT ADD CONSTRAINT DCU_LVR_TEMPS_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE  DCU_LVR_TEMPS_DAT ADD CONSTRAINT DCU_LVR_TEMPS_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);



CREATE TABLE DCU_LVRB_TEMPS_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10),  -- TT
  t1           		BINARY_FLOAT,
  t2           		BINARY_FLOAT,
  t3           		BINARY_FLOAT
);
 
ALTER TABLE  DCU_LVRB_TEMPS_DAT ADD CONSTRAINT DCU_LVRB_TEMPS_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE  DCU_LVRB_TEMPS_DAT ADD CONSTRAINT DCU_LVRB_TEMPS_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);




CREATE TABLE DCU_LVR_VOLTAGES_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10), -- TT
  VFE1_A     	      	BINARY_FLOAT,
  VFE2_A           	BINARY_FLOAT,
  VFE3_A           	BINARY_FLOAT,
  VFE4_A           	BINARY_FLOAT,
  VFE5_A           	BINARY_FLOAT,
  VCC              	BINARY_FLOAT,
  VFE4_5_D         	BINARY_FLOAT,
  VFE1_2_3_D      	BINARY_FLOAT,
  BUFFER           	BINARY_FLOAT,
  FENIX            	BINARY_FLOAT,
  V43_A            	BINARY_FLOAT,
  OCM              	BINARY_FLOAT,
  GOH              	BINARY_FLOAT,
  INH              	BINARY_FLOAT,
  V43_D            	BINARY_FLOAT
);
 
ALTER TABLE  DCU_LVR_VOLTAGES_DAT ADD CONSTRAINT DCU_LVR_VOLTAGES_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE  DCU_LVR_VOLTAGES_DAT ADD CONSTRAINT DCU_LVR_VOLTAGES_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);

CREATE TABLE DCU_CCS_DAT (
  iov_id                NUMBER(10),
  logic_id              NUMBER(10), -- TT
  M1_VDD1               BINARY_FLOAT,
  M2_VDD1               BINARY_FLOAT,
  M1_VDD2               BINARY_FLOAT,
  M2_VDD2               BINARY_FLOAT,
  M1_Vinj               BINARY_FLOAT,
  M2_Vinj               BINARY_FLOAT
  M1_VCC                BINARY_FLOAT,
  M2_VCC                BINARY_FLOAT,
  M1_DCUTemp            BINARY_FLOAT,
  M2_DCUTemp            BINARY_FLOAT,
  CCSTempLow            BINARY_FLOAT,  
  CCSTempHigh           BINARY_FLOAT
);

ALTER TABLE  DCU_CCS_DAT ADD CONSTRAINT DCU_CCS_pk PRIMARY KEY (iov_id, logic_id);
ALTER TABLE  DCU_CCS_DAT ADD CONSTRAINT DCU_CCS_fk FOREIGN KEY (iov_id) REFERENCES dcu_iov (iov_id);


