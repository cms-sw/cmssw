-- creates the tables for the ecal daq configuration 
-- 12-1-2007
-- by F. Cavallari and P. Musella
-- updated by FC on 14/3/2008
-- ********** ECAL_RUN 


CREATE TABLE ECAL_RUN_MODE_DEF (
	DEF_ID  NUMBER NOT NULL
     , RUN_MODE_STRING VARCHAR2(20)
);
ALTER TABLE ECAL_RUN_MODE_DEF ADD CONSTRAINT ecal_run_mode_def_pk PRIMARY KEY (def_id);
ALTER TABLE ECAL_RUN_MODE_DEF ADD CONSTRAINT ecal_run_mode_def_uk1 UNIQUE (run_mode_string);

CREATE SEQUENCE ecal_run_mode_def_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_run_mode_def_trg
before insert on ECAL_RUN_MODE_DEF
for each row
begin
select ecal_run_mode_def_sq.NextVal into :new.def_id from dual;
end;
/

prompt FUNCTION get_run_mode_def_id;
create or replace function get_run_mode_def_id( run_mode IN VARCHAR ) return NUMBER
IS
 	ret NUMBER;
BEGIN
	SELECT DEF_ID 
		INTO 	ret 
		FROM 	ECAL_RUN_MODE_DEF 
		WHERE 	RUN_MODE_STRING=run_mode
	;
	return (ret);
END;
/

CREATE TABLE ECAL_RUN_CONFIGURATION_DAT (
       CONFIG_ID NUMBER NOT NULL
     , TAG VARCHAR2(64) NOT NULL
     , VERSION NUMBER(22) NOT NULL
     , RUN_TYPE_DEF_ID NUMBER NOT NULL
     , RUN_MODE_DEF_ID NUMBER NOT NULL
     , NUM_OF_SEQUENCES NUMBER(22) NULL
     , DESCRIPTION VARCHAR2(200) NULL
     , DEFAULTS NUMBER NULL
     , TRG_MODE VARCHAR2(64) NULL
     , NUM_OF_EVENTS NUMber NULL
     , db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL     
     , usage_status varchar2(15) DEFAULT 'VALID'     
);

ALTER TABLE ECAL_RUN_CONFIGURATION_DAT ADD CONSTRAINT ecal_config_pk PRIMARY KEY (config_id);
ALTER TABLE ECAL_RUN_CONFIGURATION_DAT ADD CONSTRAINT ecal_config_uk1 UNIQUE (tag, version);
ALTER TABLE ECAL_RUN_CONFIGURATION_DAT ADD CONSTRAINT ecal_config_fk2 FOREIGN KEY (run_mode_def_id) REFERENCES ECAL_RUN_MODE_DEF (DEF_ID) ;

CREATE SEQUENCE ecal_run_sq INCREMENT BY 1 START WITH 1;


prompt FUNCTION get_run_conf_id;
create or replace function get_run_conf_id( the_tag IN VARCHAR, the_version in NUMBER ) return NUMBER
IS
 	ret NUMBER;
BEGIN
	SELECT CONFIG_ID 
		INTO 	ret 
		FROM 	ECAL_RUN_CONFIGURATION_DAT r 
		WHERE 	r.TAG=the_tag
		AND	r.VERSION=the_version
	;
	return (ret);
END;
/

-- ********** ECAL_SEQUENCE

CREATE TABLE ECAL_SEQUENCE_TYPE_DEF (
	DEF_ID  NUMBER NOT NULL
	, RUN_TYPE_DEF_ID NUMBER NOT NULL
        , SEQUENCE_TYPE_STRING VARCHAR2(20)
);
ALTER TABLE ECAL_SEQUENCE_TYPE_DEF ADD CONSTRAINT ecal_sequence_type_def_pk  PRIMARY KEY (def_id);
ALTER TABLE ECAL_SEQUENCE_TYPE_DEF ADD CONSTRAINT ecal_sequence_type_def_uk1 UNIQUE (run_type_def_id,sequence_type_string);

CREATE SEQUENCE ecal_sequence_type_def_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_sequence_type_def_trg
before insert on ECAL_SEQUENCE_TYPE_DEF
for each row
begin
select ecal_sequence_type_def_sq.NextVal into :new.def_id from dual;
end;
/

prompt FUNCTION get_sequence_type_def_id;
CREATE OR REPLACE FUNCTION get_sequence_type_def_id( a_run_type IN VARCHAR, seq_type VARCHAR ) return NUMBER
IS
	ret NUMBER;
BEGIN
	SELECT s.DEF_ID 
		INTO 	ret 
		FROM 	ECAL_SEQUENCE_TYPE_DEF s
			, RUN_TYPE_DEF r
		WHERE 	s.SEQUENCE_TYPE_STRING=seq_type
			AND r.RUN_TYPE=a_run_type
			AND s.RUN_TYPE_DEF_ID=r.DEF_ID
	;
	return (ret);
END;
/

CREATE TABLE ECAL_SEQUENCE_DAT (
       SEQUENCE_ID NUMBER NOT NULL
     , ECAL_CONFIG_ID NUMBER NOT NULL
     , SEQUENCE_NUM NUMBER(22) NOT NULL
     , NUM_OF_CYCLES NUMBER(22) NULL
     , SEQUENCE_TYPE_DEF_ID NUMBER NOT NULL
     , DESCRIPTION VARCHAR2(200) NULL     
);

ALTER TABLE ECAL_SEQUENCE_DAT ADD CONSTRAINT ecal_sequence_dat_pk PRIMARY KEY (sequence_id);
ALTER TABLE ECAL_SEQUENCE_DAT ADD CONSTRAINT ecal_sequence_dat_fk1 FOREIGN KEY (ecal_config_id)       REFERENCES ECAL_RUN_CONFIGURATION_DAT (CONFIG_ID);
ALTER TABLE ECAL_SEQUENCE_DAT ADD CONSTRAINT ecal_sequence_dat_fk2 FOREIGN KEY (sequence_type_def_id) REFERENCES ECAL_SEQUENCE_TYPE_DEF (DEF_ID);
ALTER TABLE ECAL_SEQUENCE_DAT ADD CONSTRAINT ecal_sequence_dat_uk1 UNIQUE (ecal_config_id, SEQUENCE_NUM);

CREATE SEQUENCE ecal_sequence_dat_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_sequence_dat_trg
before insert on ECAL_SEQUENCE_DAT
for each row
begin
select ecal_sequence_dat_sq.NextVal into :new.sequence_id from dual;
end;
/

prompt FUNCTION get_sequence_id;
create or replace function get_sequence_id( the_run_tag IN VARCHAR, the_run_version in NUMBER, the_seq_num in NUMBER ) return NUMBER
IS
 	ret NUMBER;
BEGIN
	SELECT s.SEQUENCE_ID 
		INTO 	ret 
		FROM 	ECAL_RUN_CONFIGURATION_DAT r
			, ECAL_SEQUENCE_DAT s 
		WHERE 	r.TAG=the_run_tag
		AND	r.VERSION=the_run_version
		AND	r.CONFIG_ID=s.ECAL_CONFIG_ID
		AND	s.sequence_num=the_seq_num
	;
	return (ret);
END;
/
show errors;

-- TODO Add a trigger to check that sequence type and run_type are coherent


-- ********** ECAL_CYCLE_DAT

CREATE TABLE ECAL_CYCLE_DAT (
	  CYCLE_ID NUMBER NOT NULL
	, SEQUENCE_ID NUMBER NOT NULL
	, CYCLE_NUM NUMBER(22)
        , TAG VARCHAR2(64) NULL
        , DESCRIPTION VARCHAR2(200) NULL
 );    

ALTER TABLE ECAL_CYCLE_DAT ADD CONSTRAINT ecal_cycle_dat_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_CYCLE_DAT ADD CONSTRAINT ecal_cycxle_uk1 UNIQUE (sequence_id, cycle_num);
ALTER TABLE ECAL_CYCLE_DAT ADD CONSTRAINT ecal_cycle_dat_fk1 FOREIGN KEY (sequence_id) REFERENCES ECAL_SEQUENCE_DAT (SEQUENCE_ID) ;

CREATE SEQUENCE ecal_cycle_dat_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_cycle_dat_trg
before insert on ECAL_CYCLE_DAT
for each row
begin
select ecal_cycle_dat_sq.NextVal into :new.cycle_id from dual;
end;
/

-- ********** ECAL_CCS

CREATE TABLE ECAL_CCS_CONFIGURATION (
	ccs_configuration_id NUMBER NOT NULL
        , ccs_tag VARCHAR2(32) NOT NULL
	, DACCAL NUMBER
        , DELAY NUMBER
        , GAIN VARCHAR2(64)
        , MEMGAIN VARCHAR2(64)
        , OFFSET_HIGH NUMBER
        , OFFSET_LOW  NUMBER
        , OFFSET_MID  NUMBER
	, TRG_MODE VARCHAR2(64)
        , TRG_FILTER VARCHAR2(64)
        , CLOCK NUMBER
        , BGO_SOURCE VARCHAR2(64)
        , TTS_MASK NUMBER
        , DAQ_BCID_PRESET NUMBER
        , TRIG_BCID_PRESET NUMBER
        , BC0_COUNTER NUMBER
        , BC0_DELAY NUMBER
        , TE_DELAY NUMBER
);
ALTER TABLE ECAL_CCS_CONFIGURATION ADD CONSTRAINT ecal_ccs_config_pk PRIMARY KEY (ccs_configuration_id);

CREATE SEQUENCE ecal_CCS_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_CCS_CYCLE (
	  CYCLE_ID NUMBER NOT NULL,
	  CCS_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_CCS_CYCLE ADD CONSTRAINT ecal_ccs_cycle_pk PRIMARY KEY (CYCLE_ID);
ALTER TABLE ECAL_CCS_CYCLE ADD CONSTRAINT ecal_ccs_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_CCS_CYCLE ADD CONSTRAINT ecal_ccs_cycle_fk2 FOREIGN KEY (ccs_configuration_id) REFERENCES ECAL_CCS_CONFIGURATION (ccs_configuration_id);


-- ********** ECAL_DCC

CREATE TABLE ECAL_DCC_CONFIGURATION (
	DCC_CONFIGURATION_ID NUMBER NOT NULL
        , dcc_tag VARCHAR2(32) NOT NULL
	, DCC_CONFIGURATION_URL VARCHAR2(100)
	, TESTPATTERN_FILE_URL VARCHAR2(100)
	, N_TESTPATTERNS_TO_LOAD NUMBER
        , SM_HALF NUMBER
    	, dcc_CONFIGURATION CLOB
);
ALTER TABLE ECAL_DCC_CONFIGURATION ADD CONSTRAINT ecal_dcc_config_pk PRIMARY KEY (dcc_configuration_id);

CREATE SEQUENCE ecal_DCC_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_DCC_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , DCC_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_DCC_CYCLE ADD CONSTRAINT ecal_dcc_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_DCC_CYCLE ADD CONSTRAINT ecal_dcc_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_DCC_CYCLE ADD CONSTRAINT ecal_dcc_cycle_fk2 FOREIGN KEY (dcc_configuration_id) REFERENCES ECAL_DCC_CONFIGURATION (dcc_configuration_id);


-- ********** ECAL_DCu

CREATE TABLE ECAL_DCu_CONFIGURATION (
	DCu_CONFIGURATION_ID NUMBER NOT NULL
        , dcu_tag VARCHAR2(32) NOT NULL
);
ALTER TABLE ECAL_DCu_CONFIGURATION ADD CONSTRAINT ecal_dcu_config_pk PRIMARY KEY (dcu_configuration_id);

CREATE SEQUENCE ecal_DCu_CONFIG_sq INCREMENT BY 1 START WITH 1;
CREATE TABLE ECAL_DCU_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , DCU_CONFIGURATION_ID NUMBER NOT NULL
         );    
ALTER TABLE ECAL_DCU_CYCLE ADD CONSTRAINT ecal_dcu_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_DCU_CYCLE ADD CONSTRAINT ecal_dcu_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_DCU_CYCLE ADD CONSTRAINT ecal_dcu_cycle_fk2 FOREIGN KEY (dcu_configuration_id) REFERENCES ECAL_DCu_CONFIGURATION (dcu_configuration_id);



-- ********** ECAL_ttcf

CREATE TABLE ECAL_TTCF_CONFIGURATION (
	TTCF_CONFIGURATION_ID NUMBER NOT NULL
        , TTCF_tag VARCHAR2(32) NOT NULL
        , TTCF_CONFIGURATION_FILE VARCHAR2(100) 
      	, TTCF_CONFIGURATION CLOB,
RXBC0_DELAY NUMBER, REG_30 NUMBER
);
ALTER TABLE ECAL_TTCF_CONFIGURATION ADD CONSTRAINT ecal_ttcf_config_pk PRIMARY KEY (ttcf_configuration_id);

CREATE SEQUENCE ecal_TTCF_CONFIG_sq INCREMENT BY 1 START WITH 1;

CREATE TABLE ECAL_TTCF_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , TTCF_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_TTCF_CYCLE ADD CONSTRAINT ecal_ttcf_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_TTCF_CYCLE ADD CONSTRAINT ecal_ttcf_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_TTCF_CYCLE ADD CONSTRAINT ecal_ttcf_cycle_fk2 FOREIGN KEY (ttcf_configuration_id) REFERENCES ECAL_TTCF_CONFIGURATION (ttcf_configuration_id);


-- ********** ECAL_srp

CREATE TABLE ECAL_SRP_CONFIGURATION (
	SRP_CONFIGURATION_ID NUMBER NOT NULL
        , SRP_tag VARCHAR2(32) NOT NULL
	, DEBUGMODE NUMBER
	, DUMMYMODE NUMBER
	, PATTERN_DIRECTORY VARCHAR2(100)
        , AUTOMATIC_MASKS NUMBER
        , SRP0BUNCHADJUSTPOSITION NUMBER 
	, SRP_CONFIG_FILE VARCHAR2(100)
      	, SRP_CONFIGURATION CLOB
);
ALTER TABLE ECAL_SRP_CONFIGURATION ADD CONSTRAINT ecal_SRP_config_pk PRIMARY KEY (SRP_configuration_id);

CREATE SEQUENCE ecal_SRP_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_SRP_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , srp_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_srp_CYCLE ADD CONSTRAINT ecal_srp_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_srp_CYCLE ADD CONSTRAINT ecal_srp_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_srp_CYCLE ADD CONSTRAINT ecal_srp_cycle_fk2 FOREIGN KEY (srp_configuration_id) REFERENCES ECAL_SRP_CONFIGURATION (srp_configuration_id);


-- ********** ECAL_LASER

CREATE TABLE ECAL_LASER_CONFIGURATION (
	LASER_configuration_id NUMBER NOT NULL
        , laser_tag VARCHAR2(32) NOT NULL
        , laser_DEBUG NUMBER
        , DUMMY NUMBER
-- ********** ECAL_MATACQ
        , MATACQ_BASE_ADDRESS NUMBER
        , MATACQ_NONE NUMBER
	, matacq_mode VARCHAR2(64)
        , channel_Mask NUMBER
        , max_Samples_For_Daq VARCHAR2(64)
        , maTACQ_FED_ID NUMBER
        , pedestal_File VARCHAR2(128)
        , use_Buffer NUMBER
        , postTrig NUMBER
        , fp_Mode NUMBER
        , hal_Module_File VARCHAR2(64)
        , hal_Address_Table_File VARCHAR2(64)
        , hal_Static_Table_File VARCHAR2(64)
        , matacq_Serial_Number VARCHAR2(64)
        , pedestal_Run_Event_Count NUMBER
        , raw_Data_Mode NUMBER
        , ACQUISITION_MODE VARCHAR2(64)
        , LOCAL_OUTPUT_FILE VARCHAR2(100)
	, MATACQ_VERNIER_MIN NUMBER
	, MATACQ_VERNIER_MAX NUMBER
-- *********** emtc
        , emtc_none NUMBER
        , wte2_laser_delay NUMBER
        , laser_phase NUMBER
        , emtc_ttc_in NUMBER
        , emtc_slot_id NUMBER
-- *********** ecal laser
 	, WAVELENGTH NUMBER
	, POWER_SETTING NUMBER
      	, OPTICAL_SWITCH NUMBER
      	, FILTER NUMBER
	, LASER_CONTROL_ON NUMBER
	, LASER_CONTROL_HOST VARCHAR2(32)
	, LASER_CONTROL_PORT NUMBER
        , laser_tag2 varchar2(32)
, wte_2_led_delay NUMBER(4)
, led1_on NUMBER(1)
, led2_on NUMBER(1)
, led3_on NUMBER(1)
, led4_on NUMBER(1)
, VINJ NUMBER
, orange_led_mon_ampl number
, blue_led_mon_ampl number
, trig_log_file varchar2(512)
, led_control_on NUMBER(1)
, led_control_host varchar2(100)
, led_control_port NUMBER(5)
, ir_laser_power number(3)
, green_laser_power number(3)
, red_laser_power number(3)
, blue_laser_log_attenuator number(3) 
, IR_LASER_LOG_ATTENUATOR NUMBER(3)
, GREEN_LASER_LOG_ATTENUATOR  NUMBER(3)
, RED_LASER_LOG_ATTENUATOR NUMBER(3)
, LASER_CONFIG_FILE VARCHAR2(512)
, laser_configuration CLOB
);

ALTER TABLE ECAL_LASER_CONFIGURATION ADD CONSTRAINT ecal_LASER_config_pk PRIMARY KEY (LASER_configuration_id);

CREATE SEQUENCE ecal_LASER_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_LASER_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , LASER_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_LASER_CYCLE ADD CONSTRAINT ecal_LASER_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_LASER_CYCLE ADD CONSTRAINT ecal_LASER_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_LASER_CYCLE ADD CONSTRAINT ecal_LASER_cycle_fk2 FOREIGN KEY (laser_configuration_id) REFERENCES ECAL_LASER_CONFIGURATION (LASER_configuration_id);


-- ********** ECAL_TCC

CREATE TABLE ECAL_TCC_CONFIGURATION (
	TCC_configuration_id NUMBER NOT NULL
        , TCC_tag VARCHAR2(32) NOT NULL
	, Configuration_file varchar2(100) NULL
        , LUT_CONFIGURATION_FILE VARCHAR2(100) NULL
        , SLB_CONFIGURATION_FILE VARCHAR2(100) NULL
        , TESTPATTERNFILE_URL VARCHAR2(100) NULL
        , N_TESTPATTERNS_TO_LOAD number NULL
               , tcc_configuration CLOB 
               , lut_configuration CLOB 
               , slb_configuration CLOB 
);
ALTER TABLE ECAL_TCC_CONFIGURATION ADD CONSTRAINT ecal_TCC_config_pk PRIMARY KEY (TCC_configuration_id);

CREATE SEQUENCE ecal_TCC_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_TCC_CYCLE (
 CYCLE_ID NUMBER NOT NULL
	 , TCC_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_TCC_CYCLE ADD CONSTRAINT ecal_TCC_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_TCC_CYCLE ADD CONSTRAINT ecal_TCC_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_TCC_CYCLE ADD CONSTRAINT ecal_TCC_cycle_fk2 FOREIGN KEY (tcc_configuration_id) REFERENCES ECAL_TCC_CONFIGURATION (TCC_configuration_id);


-- ********** ECAL_TTCci

CREATE TABLE ECAL_TTCCI_CONFIGURATION (
	TTCCI_configuration_id NUMBER NOT NULL
        , TTCCI_tag VARCHAR2(32) NOT NULL
	, TTCCI_configuration_file varchar2(130)
	,TRG_MODE varchar2(32)
	,TRG_SLEEP NUMBER
    	, Configuration CLOB
       , CONFIGURATION_SCRIPT varchar2(100)
       , CONFIGURATION_SCRIPT_PARAMS varchar2(100)
);
ALTER TABLE ECAL_TTCCI_CONFIGURATION ADD CONSTRAINT ecal_TTCCI_config_pk PRIMARY KEY (TTCCI_configuration_id);

CREATE SEQUENCE ecal_TTCCI_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_TTCCI_CYCLE (
           CYCLE_ID NUMBER NOT NULL
	 , TTCCI_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_TTCCI_CYCLE ADD CONSTRAINT ecal_TTCCI_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_TTCCI_CYCLE ADD CONSTRAINT ecal_TTCCI_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_TTCCI_CYCLE ADD CONSTRAINT ecal_TTCCI_cycle_fk2 FOREIGN KEY (ttcci_configuration_id) REFERENCES ECAL_TTCCI_CONFIGURATION (TTCCI_configuration_id);



-- ********** ECAL_LTC

CREATE TABLE ECAL_LTC_CONFIGURATION (
	LTC_configuration_id NUMBER NOT NULL
        , LTC_tag VARCHAR2(32) NOT NULL
	, ltc_Configuration_file varchar2(100)
      	, Configuration CLOB
);
ALTER TABLE ECAL_LTC_CONFIGURATION ADD CONSTRAINT ecal_LTC_config_pk PRIMARY KEY (LTC_configuration_id);

CREATE SEQUENCE ecal_LTC_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_LTC_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , LTC_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_LTC_CYCLE ADD CONSTRAINT ecal_LTC_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_LTC_CYCLE ADD CONSTRAINT ecal_LTC_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_LTC_CYCLE ADD CONSTRAINT ecal_LTC_cycle_fk2 FOREIGN KEY (ltc_configuration_id) REFERENCES ECAL_LTC_CONFIGURATION (LTC_configuration_id);

-- ********** ECAL_LTS

CREATE TABLE ECAL_LTS_CONFIGURATION (
	LTS_configuration_id NUMBER NOT NULL
        , lts_tag VARCHAR2(32) NOT NULL
        , TRIGGER_TYPE VARCHAR2(32)
        , NUM_OF_EVENTS NUMBER
        , RATE NUMBER
	, TRIG_LOC_L1_DELAY NUMBER
);
ALTER TABLE ECAL_LTS_CONFIGURATION ADD CONSTRAINT ecal_LTS_config_pk PRIMARY KEY (LTS_configuration_id);

CREATE SEQUENCE ecal_LTS_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_LTS_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , LTS_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_LTS_CYCLE ADD CONSTRAINT ecal_LTS_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_LTS_CYCLE ADD CONSTRAINT ecal_LTS_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_LTS_CYCLE ADD CONSTRAINT ecal_LTS_cycle_fk2 FOREIGN KEY (lts_configuration_id) REFERENCES ECAL_LTS_CONFIGURATION (LTS_configuration_id);

-- ********** ECAL_JBH4

CREATE TABLE ECAL_JBH4_CONFIGURATION (
	JBH4_configuration_id NUMBER NOT NULL
        , JBH4_tag VARCHAR2(32) NOT NULL
        , useBuffer NUMBER
        , halModuleFile VARCHAR2(64)
        , halAddressTableFile VARCHAR2(64)
        , halStaticTableFile VARCHAR2(64)
        , halCbd8210SerialNumber VARCHAR2(64)
        , caenBridgeType VARCHAR2(64)
        , caenLinkNumber NUMBER
        , caenBoardNumber NUMBER
);
ALTER TABLE ECAL_JBH4_CONFIGURATION ADD CONSTRAINT ecal_JBH4_config_pk PRIMARY KEY (JBH4_configuration_id);

CREATE SEQUENCE ecal_JBH4_CONFIG_sq INCREMENT BY 1 START WITH 1;


CREATE TABLE ECAL_JBH4_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , JBH4_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_JBH4_CYCLE ADD CONSTRAINT ecal_JBH4_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_JBH4_CYCLE ADD CONSTRAINT ecal_JBH4_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_JBH4_CYCLE ADD CONSTRAINT ecal_JBH4_cycle_fk2 FOREIGN KEY (jbh4_configuration_id) REFERENCES ECAL_JBH4_CONFIGURATION (JBH4_configuration_id);


-- ********** ECAL_SCAN

CREATE TABLE ECAL_SCAN_DAT (
          SCAN_ID NUMBER	
        , SCAN_tag VARCHAR2(32) NOT NULL
        , type_id number
        , scan_type varchar2(32) 
	, FROM_VAL NUMBER
	, TO_VAL NUMBER
	, STEP NUMBER
);

ALTER TABLE ECAL_SCAN_DAT ADD CONSTRAINT ecal_scan_dat_pk  PRIMARY KEY (scan_id);
CREATE SEQUENCE ecal_SCAN_CONFIG_sq INCREMENT BY 1 START WITH 1;





CREATE TABLE ECAL_SCAN_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , SCAN_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_SCAN_CYCLE ADD CONSTRAINT ecal_SCAN_cycle_pk  PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_SCAN_CYCLE ADD CONSTRAINT ecal_SCAN_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_SCAN_CYCLE ADD CONSTRAINT ecal_SCAN_cycle_fk2 FOREIGN KEY (scan_id)  REFERENCES ECAL_SCAN_DAT  (SCAN_id);

-- ********** VIEWS

CREATE OR REPLACE VIEW ECAL_RUN_CONFIGURATION AS
select r.CONFIG_ID
     , r.TAG
     , r.VERSION
     , rtd.RUN_TYPE RUN_TYPE
     , rmd.RUN_MODE_STRING RUN_MODE
     , r.NUM_OF_SEQUENCES
     , r.DESCRIPTION RUN_CONFIG_DESCRIPTION
     , r.DEFAULTS
     , r.TRG_MODE
     , r.usage_status 	
from
	ECAL_RUN_CONFIGURATION_DAT r
	, RUN_TYPE_DEF rtd
	, ECAL_RUN_MODE_DEF rmd
where
	r.RUN_TYPE_DEF_ID=rtd.DEF_ID
	and r.RUN_MODE_DEF_ID=rmd.DEF_ID
;


CREATE OR REPLACE VIEW ECAL_SEQUENCE AS
select
       s.SEQUENCE_ID
     , r.TAG
     , r.VERSION
     , s.SEQUENCE_NUM
     , s.NUM_OF_CYCLES
     , std.SEQUENCE_TYPE_STRING sequence_type
     , s.DESCRIPTION
from
	ECAL_SEQUENCE_DAT s
	, ECAL_SEQUENCE_TYPE_DEF std
	, ECAL_RUN_CONFIGURATION_DAT r
where
	s.ECAL_CONFIG_ID=r.CONFIG_ID
	and s.SEQUENCE_TYPE_DEF_ID=std.DEF_ID
;

CREATE OR REPLACE VIEW ECAL_CYCLE AS
SELECT 
	e.cycle_id
	, r.tag tag
	, r.version version
	, s.sequence_num
        , s.sequence_id
	, e.cycle_num
	, e.tag cycle_tag
        , e.description
	, ccs.CCS_CONFIGURATION_ID
	, dcc.dcc_CONFIGURATION_ID
	, laser.laser_CONFIGURATION_ID
	, ltc.ltc_CONFIGURATION_ID
	, lts.lts_CONFIGURATION_ID
	, dcu.dcu_CONFIGURATION_ID
	, tcc.tcc_CONFIGURATION_ID
        , ttcf.ttcf_CONFIGURATION_ID
        , srp.srp_configuration_id
	, ttcci.ttcci_CONFIGURATION_ID "TTCci_CONFIGURATION_ID"
	, jbh4.jbh4_CONFIGURATION_ID
	, scan.scan_id
FROM
	ECAL_RUN_CONFIGURATION_DAT r,
	ECAL_SEQUENCE_DAT s,
	ECAL_CYCLE_Dat e
	LEFT OUTER join
	ECAL_CCS_CYCLE ccs on  e.cycle_id=ccs.cycle_ID
	left outer join
	ECAL_DCC_CYCLE dcc on  e.cycle_id=dcc.cycle_ID
	left outer join
	ECAL_LASER_CYCLE laser on e.cycle_id=laser.cycle_ID
	left outer join
	ECAL_ltc_CYCLE ltc on  e.cycle_id=ltc.cycle_ID
	left outer join
	ECAL_lts_CYCLE lts on e.cycle_id=lts.cycle_ID
	left outer join
	ECAL_dcu_CYCLE dcu on e.cycle_id=dcu.cycle_ID
	left outer join
	ECAL_tcc_CYCLE tcc on e.cycle_id=tcc.cycle_ID
	left outer join
	ECAL_ttcci_CYCLE ttcci on  e.cycle_id=ttcci.cycle_ID
	left outer join
	ECAL_jbh4_CYCLE jbh4 on  e.cycle_id=jbh4.cycle_ID
	left outer join
	ECAL_SCAN_cycle scan on e.cycle_id=scan.cycle_id
	left outer join
	ECAL_srp_cycle srp on e.cycle_id=srp.cycle_id
	left outer join
	ECAL_ttcf_CYCLE ttcf on e.cycle_id=ttcf.cycle_ID
where 
	r.config_id=s.ecal_config_id 
	and e.sequence_id=s.sequence_id
;


CREATE OR REPLACE VIEW ECAL_SCAN_CONFIGURATION AS
select r.SCAN_ID SCAN_ID
        , r.SCAN_tag tag
     , r.type_id type_id
     , r.scan_type scan_type
     , r.FROM_VAL from_val
     , r.to_val to_val
     , r.STEP  step
 from
	ECAL_scan_dat r
;





@insert_run_mod_defs
