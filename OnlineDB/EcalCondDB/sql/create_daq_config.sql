-- creates the tables for the ecal daq configuration 
-- 12-1-2007
-- by F. Cavallari and P. Musella
-- updated by FC on 14/3/2008
-- ********** ECAL_RUN 

CREATE TABLE RUN_TYPE_DEF (
	DEF_ID  NUMBER NOT NULL
     , RUN_TYPE VARCHAR2(20),
        DESCRIPTION VARCHAR2(100)
);
ALTER TABLE RUN_TYPE_DEF ADD CONSTRAINT run_type_def_pk PRIMARY KEY (def_id);
ALTER TABLE RUN_TYPE_DEF ADD CONSTRAINT run_type_def_uk1 UNIQUE (run_type);

CREATE SEQUENCE run_type_def_sq INCREMENT BY 1 START WITH 1;
CREATE trigger run_type_def_trg
before insert on RUN_TYPE_DEF
for each row
begin
select run_type_def_sq.NextVal into :new.def_id from dual;
end;
/

prompt FUNCTION get_run_type_def_id;
create or replace function get_run_type_def_id( a_run_type IN VARCHAR ) return NUMBER
IS
 	ret NUMBER;
BEGIN
	SELECT DEF_ID 
		INTO 	ret 
		FROM 	RUN_TYPE_DEF 
		WHERE 	RUN_TYPE=a_run_type
	;
	return (ret);
END;
/

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
     , DESCRIPTION VARCHAR2(200) NULL,
db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL     
);

ALTER TABLE ECAL_RUN_CONFIGURATION_DAT ADD CONSTRAINT ecal_config_pk PRIMARY KEY (config_id);
ALTER TABLE ECAL_RUN_CONFIGURATION_DAT ADD CONSTRAINT ecal_config_uk1 UNIQUE (tag, version);
ALTER TABLE ECAL_RUN_CONFIGURATION_DAT ADD CONSTRAINT ecal_config_fk1 FOREIGN KEY (run_type_def_id) REFERENCES RUN_TYPE_DEF (DEF_ID) ;
ALTER TABLE ECAL_RUN_CONFIGURATION_DAT ADD CONSTRAINT ecal_config_fk2 FOREIGN KEY (run_mode_def_id) REFERENCES ECAL_RUN_MODE_DEF (DEF_ID) ;

CREATE SEQUENCE ecal_run_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_run_trg
before insert on ECAL_RUN_CONFIGURATION_DAT
for each row
begin
select ecal_run_sq.NextVal into :new.config_id from dual;
end;
/

show errors;

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
ALTER TABLE ECAL_SEQUENCE_TYPE_DEF ADD CONSTRAINT ecal_sequence_type_def_fk1 FOREIGN KEY (run_type_def_id) REFERENCES  RUN_TYPE_DEF (DEF_ID) ;
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
	, DACCAL NUMBER
        , DELAY NUMBER
        , GAIN VARCHAR2(64)
        , MEMGAIN VARCHAR2(64)
        , OFFSET_HIGH NUMBER
        , OFFSET_LOW  NUMBER
        , OFFSET_MID  NUMBER
        , PEDESTAL_OFFSET_RELEASE VARCHAR2(64)
        , SYSTEM  VARCHAR2(64)
	, TRG_MODE VARCHAR2(64)
        , TRG_FILTER VARCHAR2(64)
);
ALTER TABLE ECAL_CCS_CONFIGURATION ADD CONSTRAINT ecal_ccs_config_pk PRIMARY KEY (ccs_configuration_id);

CREATE SEQUENCE ecal_CCS_CONFIG_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_CCS_CONFIG_trg
before insert on ECAL_CCS_CONFIGURATION
for each row
begin
select ecal_CCS_CONFIG_sq.NextVal into :new.CCS_configuration_id from dual;
end;
/


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
	, DCC_CONFIGURATION CLOB
);
ALTER TABLE ECAL_DCC_CONFIGURATION ADD CONSTRAINT ecal_dcc_config_pk PRIMARY KEY (dcc_configuration_id);

CREATE SEQUENCE ecal_DCC_CONFIG_sq INCREMENT BY 1 START WITH 1;
-- CREATE trigger ecal_DCC_CONFIG_trg
-- before insert on ECAL_DCC_CONFIGURATION
-- for each row
-- begin
-- select ecal_DCC_CONFIG_sq.NextVal into :new.dcc_configuration_id from dual;
-- end;
-- /


CREATE TABLE ECAL_DCC_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , DCC_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_DCC_CYCLE ADD CONSTRAINT ecal_dcc_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_DCC_CYCLE ADD CONSTRAINT ecal_dcc_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_DCC_CYCLE ADD CONSTRAINT ecal_dcc_cycle_fk2 FOREIGN KEY (dcc_configuration_id) REFERENCES ECAL_DCC_CONFIGURATION (dcc_configuration_id);


-- ********** ECAL_LASER

CREATE TABLE ECAL_LASER_CONFIGURATION (
	LASER_configuration_id NUMBER NOT NULL
 	, WAVELENGTH NUMBER
	, POWER_SETTING NUMBER
      	, OPTICAL_SWITCH NUMBER
	, FILTER NUMBER
);
ALTER TABLE ECAL_LASER_CONFIGURATION ADD CONSTRAINT ecal_LASER_config_pk PRIMARY KEY (LASER_configuration_id);

CREATE SEQUENCE ecal_LASER_CONFIG_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_LASER_CONFIG_trg
before insert on ECAL_LASER_CONFIGURATION
for each row
begin
select ecal_LASER_CONFIG_sq.NextVal into :new.LASER_configuration_id from dual;
end;
/


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
	, DEVICE_CONFIG_PARAM_ID NUMBER NOT NULL 
);
ALTER TABLE ECAL_TCC_CONFIGURATION ADD CONSTRAINT ecal_TCC_config_pk PRIMARY KEY (TCC_configuration_id);

CREATE SEQUENCE ecal_TCC_CONFIG_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_TCC_CONFIG_trg
before insert on ECAL_TCC_CONFIGURATION
for each row
begin
select ecal_TCC_CONFIG_sq.NextVal into :new.TCC_configuration_id from dual;
end;
/


CREATE TABLE ECAL_TCC_CYCLE (
 CYCLE_ID NUMBER NOT NULL
	 , TCC_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_TCC_CYCLE ADD CONSTRAINT ecal_TCC_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_TCC_CYCLE ADD CONSTRAINT ecal_TCC_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_TCC_CYCLE ADD CONSTRAINT ecal_TCC_cycle_fk2 FOREIGN KEY (tcc_configuration_id) REFERENCES ECAL_TCC_CONFIGURATION (TCC_configuration_id);


-- ********** ECAL_TCCci

CREATE TABLE ECAL_TTCCI_CONFIGURATION (
	TTCCI_configuration_id NUMBER NOT NULL
	, Configuration CLOB
);
ALTER TABLE ECAL_TTCCI_CONFIGURATION ADD CONSTRAINT ecal_TTCCI_config_pk PRIMARY KEY (TTCCI_configuration_id);

CREATE SEQUENCE ecal_TTCCI_CONFIG_sq INCREMENT BY 1 START WITH 1;
-- CREATE trigger ecal_TTCCI_CONFIG_trg
-- before insert on ECAL_TTCCI_CONFIGURATION
-- for each row
-- begin
-- select ecal_TTCCI_CONFIG_sq.NextVal into :new.TTCCI_configuration_id from dual;
-- end;
-- /


CREATE TABLE ECAL_TTCCI_CYCLE (
           CYCLE_ID NUMBER NOT NULL
	 , TTCCI_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_TTCCI_CYCLE ADD CONSTRAINT ecal_TTCCI_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_TTCCI_CYCLE ADD CONSTRAINT ecal_TTCCI_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_TTCCI_CYCLE ADD CONSTRAINT ecal_TTCCI_cycle_fk2 FOREIGN KEY (ttcci_configuration_id) REFERENCES ECAL_TTCCI_CONFIGURATION (TTCCI_configuration_id);


-- ********** ECAL_MATACQ

CREATE TABLE ECAL_MATACQ_CONFIGURATION (
	MATACQ_configuration_id NUMBER NOT NULL
	, matacq_mode VARCHAR2(64)
        , fastPedestal NUMBER
        , channelMask NUMBER
        , maxSamplesForDaq VARCHAR2(64)
        , pedestalFile VARCHAR2(128)
        , useBuffer NUMBER
        , postTrig NUMBER
        , fpMode NUMBER
        , halModuleFile VARCHAR2(64)
        , halAddressTableFile VARCHAR2(64)
        , halStaticTableFile VARCHAR2(64)
        , matacqSerialNumber VARCHAR2(64)
        , pedestalRunEventCount NUMBER
        , rawDataMode NUMBER
);
ALTER TABLE ECAL_MATACQ_CONFIGURATION ADD CONSTRAINT ecal_MATACQ_config_pk PRIMARY KEY (MATACQ_configuration_id);

CREATE SEQUENCE ecal_MATACQ_CONFIG_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_MATACQ_CONFIG_trg
before insert on ECAL_MATACQ_CONFIGURATION
for each row
begin
select ecal_MATACQ_CONFIG_sq.NextVal into :new.MATACQ_configuration_id from dual;
end;
/


CREATE TABLE ECAL_MATACQ_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , MATACQ_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_MATACQ_CYCLE ADD CONSTRAINT ecal_MATACQ_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_MATACQ_CYCLE ADD CONSTRAINT ecal_MATACQ_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_MATACQ_CYCLE ADD CONSTRAINT ecal_MATACQ_cycle_fk2 FOREIGN KEY (matacq_configuration_id) REFERENCES ECAL_MATACQ_CONFIGURATION (MATACQ_configuration_id);

-- ********** ECAL_LTC

CREATE TABLE ECAL_LTC_CONFIGURATION (
	LTC_configuration_id NUMBER NOT NULL
	, DEVICE_CONFIG_PARAM_ID NUMBER NOT NULL 
);
ALTER TABLE ECAL_LTC_CONFIGURATION ADD CONSTRAINT ecal_LTC_config_pk PRIMARY KEY (LTC_configuration_id);

CREATE SEQUENCE ecal_LTC_CONFIG_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_LTC_CONFIG_trg
before insert on ECAL_LTC_CONFIGURATION
for each row
begin
select ecal_LTC_CONFIG_sq.NextVal into :new.LTC_configuration_id from dual;
end;
/


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
        , TRIGGER_TYPE VARCHAR2(32)
        , NUM_OF_EVENTS NUMBER
        , RATE NUMBER
	, TRIG_LOC_L1_DELAY NUMBER
);
ALTER TABLE ECAL_LTS_CONFIGURATION ADD CONSTRAINT ecal_LTS_config_pk PRIMARY KEY (LTS_configuration_id);

CREATE SEQUENCE ecal_LTS_CONFIG_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_LTS_CONFIG_trg
before insert on ECAL_LTS_CONFIGURATION
for each row
begin
select ecal_LTS_CONFIG_sq.NextVal into :new.LTS_configuration_id from dual;
end;
/


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
CREATE trigger ecal_JBH4_CONFIG_trg
before insert on ECAL_JBH4_CONFIGURATION
for each row
begin
select ecal_JBH4_CONFIG_sq.NextVal into :new.JBH4_configuration_id from dual;
end;
/

CREATE TABLE ECAL_JBH4_CYCLE (
	  CYCLE_ID NUMBER NOT NULL
	 , JBH4_CONFIGURATION_ID NUMBER NOT NULL
         );    

ALTER TABLE ECAL_JBH4_CYCLE ADD CONSTRAINT ecal_JBH4_cycle_pk PRIMARY KEY (cycle_id);
ALTER TABLE ECAL_JBH4_CYCLE ADD CONSTRAINT ecal_JBH4_cycle_fk1 FOREIGN KEY (cycle_id) REFERENCES ECAL_CYCLE_DAT (cycle_id);
ALTER TABLE ECAL_JBH4_CYCLE ADD CONSTRAINT ecal_JBH4_cycle_fk2 FOREIGN KEY (jbh4_configuration_id) REFERENCES ECAL_JBH4_CONFIGURATION (JBH4_configuration_id);


-- ********** ECAL_SCAN

CREATE TABLE ECAL_SCAN_DEF (
	DEF_ID NUMBER
	, SCAN_TYPE VARCHAR2(20)
);
ALTER TABLE ECAL_SCAN_DEF ADD CONSTRAINT ecal_SCAN_DEF_pk  PRIMARY KEY (def_id);
ALTER TABLE ECAL_SCAN_DEF ADD CONSTRAINT ecal_SCAN_DEF_uk1 UNIQUE (scan_type);
CREATE SEQUENCE ecal_scan_def_sq INCREMENT BY 1 START WITH 1;
CREATE trigger ecal_scan_def_trg
before insert on ECAL_SCAN_DEF
for each row
begin
select ecal_SCAN_DEF_sq.NextVal into :new.def_id from dual;
end;
/


CREATE TABLE ECAL_SCAN_DAT (
	SCAN_ID NUMBER
	, SCAN_DEF_ID NUMBER NOT NULL
	, DESCRIPTION VARCHAR2(100)
	, FROM_VAL NUMBER
	, TO_VAL NUMBER
	, STEP NUMBER
);
ALTER TABLE ECAL_SCAN_DAT ADD CONSTRAINT ecal_scan_dat_pk  PRIMARY KEY (scan_id);
ALTER TABLE ECAL_SCAN_DAT ADD CONSTRAINT ecal_scan_dat_fk1 FOREIGN KEY (scan_def_id) REFERENCES ECAL_scan_DEF (def_id);

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
     , r.DESCRIPTION
from
	ECAL_RUN_CONFIGURATION_DAT r
	, RUN_TYPE_DEF rtd
	, ECAL_RUN_MODE_DEF rmd
where
	r.RUN_TYPE_DEF_ID=rtd.DEF_ID
	and r.RUN_MODE_DEF_ID=rmd.DEF_ID
;

CREATE OR REPLACE TRIGGER ecal_run_configuration_insert
INSTEAD OF INSERT ON ecal_run_configuration
REFERENCING NEW AS n
FOR EACH ROW
BEGIN
	insert into ecal_run_configuration_dat(CONFIG_ID, TAG, VERSION
		, RUN_TYPE_DEF_ID, RUN_MODE_DEF_ID
		, NUM_OF_SEQUENCES, DESCRIPTION)
	values ( :n.CONFIG_ID, :n.TAG, :n.VERSION
		, get_run_type_def_id(:n.RUN_TYPE), get_run_mode_def_id(:n.RUN_MODE)
		, :n.NUM_OF_SEQUENCES, :n.DESCRIPTION
	);
END;
/
show errors;

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

CREATE OR REPLACE TRIGGER ecal_sequence_insert
INSTEAD OF INSERT ON ecal_sequence
REFERENCING NEW AS n
FOR EACH ROW
DECLARE
	conf_id NUMBER;
	seq_type_id NUMBER;
	run_type VARCHAR(20);
BEGIN
	--- get the id for the configuration we want to modify  
	conf_id := get_run_conf_id( :n.TAG, :n.VERSION);
	--- get the run type for that configuration  
	SELECT RUN_TYPE 
	INTO run_type
	FROM ECAL_RUN_CONFIGURATION WHERE CONFIG_ID=conf_id;
	seq_type_id := get_sequence_type_def_id( run_type, :n.SEQUENCE_TYPE );
	--- create the new row in the ecal_sequence_dat table
	insert into ecal_sequence_dat(SEQUENCE_ID, ECAL_CONFIG_ID, SEQUENCE_NUM, NUM_OF_CYCLES, SEQUENCE_TYPE_DEF_ID, DESCRIPTION)
		values ( :n.SEQUENCE_ID, conf_id, :n.SEQUENCE_NUM, :n.NUM_OF_CYCLES, seq_type_id, :n.DESCRIPTION
	);
END;
/
show errors;

CREATE OR REPLACE VIEW ECAL_CYCLE AS
SELECT 
	e.cycle_id
	, r.tag tag
	, r.version version
	, s.sequence_num
	, e.cycle_num
	, e.tag cycle_tag, e.description
	, ccs.CCS_CONFIGURATION_ID
	, dcc.dcc_CONFIGURATION_ID
	, laser.laser_CONFIGURATION_ID
	, ltc.ltc_CONFIGURATION_ID
	, lts.lts_CONFIGURATION_ID
	, tcc.tcc_CONFIGURATION_ID
	, ttcci.ttcci_CONFIGURATION_ID "TTCci_CONFIGURATION_ID"
	, matacq.matacq_CONFIGURATION_ID
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
	ECAL_tcc_CYCLE tcc on e.cycle_id=tcc.cycle_ID
	left outer join
	ECAL_ttcci_CYCLE ttcci on  e.cycle_id=ttcci.cycle_ID
	left outer join
	ECAL_matacq_CYCLE matacq on  e.cycle_id=matacq.cycle_ID
	left outer join
	ECAL_jbh4_CYCLE jbh4 on  e.cycle_id=jbh4.cycle_ID
	left outer join
	ECAL_SCAN_cycle scan on e.cycle_id=scan.cycle_id
where 
	r.config_id=s.ecal_config_id 
	and e.sequence_id=s.sequence_id
;

CREATE OR REPLACE TRIGGER ecal_cycle_insert
INSTEAD OF INSERT ON ecal_cycle
REFERENCING NEW AS n
FOR EACH ROW
DECLARE
	conf_id NUMBER;
	seq_id NUMBER;
	run_type VARCHAR(20);
BEGIN
	--- get the the sequence_id corrensponding to the requested run and sequence number   
	seq_id := get_sequence_id( :n.TAG, :n.VERSION, :n.SEQUENCE_NUM);
	--- create the new CYCLE
	INSERT INTO ECAL_CYCLE_DAT(CYCLE_ID, SEQUENCE_ID, CYCLE_NUM, TAG, DESCRIPTION)
	VALUES('1',seq_id,:n.CYCLE_NUM,:n.CYCLE_TAG,:n.DESCRIPTION);
-- 	IF ( :n.ccs_config_id != NULL ) THEN
-- 	END IF;
END;
/
show errors;

CREATE OR REPLACE VIEW ECAL_SCAN AS
SELECT
	s.SCAN_ID
	, sd.SCAN_TYPE
	, s.DESCRIPTION
	, s.FROM_VAL
	, s.TO_VAL
	, s.STEP
FROM
	ECAL_SCAN_DAT s
	, ECAL_SCAN_DEF sd
WHERE	
	s.SCAN_ID = sd.DEF_ID
;


-- CREATE OR REPLACE VIEW ECAL_JOINT_RUN_CONFIGURATION AS
-- SELECT  r.CONFIG_ID
-- 	, r.TAG config_tag
-- 	, r.VERSION config_version
-- 	, rtd.RUN_TYPE_STRING RUN_TYPE
-- 	, rmd.RUN_MODE_STRING RUN_MODE
-- 	, r.NUM_OF_SEQUENCES
-- 	, r.DESCRIPTION run_description
-- 	, s.SEQUENCE_NUM
-- 	, s.NUM_OF_CYCLES
-- 	, std.SEQUENCE_TYPE_STRING sequence_type
-- 	, s.DESCRIPTION sequence_description
-- 	, e.cycle_num
-- 	, e.tag cycle_tag
-- 	, e.description cycle_description
-- 	, ccs.CCS_CONFIGURATION_ID
-- 	, dcc.dcc_CONFIGURATION_ID
-- 	, laser.laser_CONFIGURATION_ID
-- 	, ltc.ltc_CONFIGURATION_ID
-- 	, lts.lts_CONFIGURATION_ID
-- 	, tcc.tcc_CONFIGURATION_ID
-- 	, ttcci.ttcci_CONFIGURATION_ID
-- 	, matacq.matacq_CONFIGURATION_ID
-- 	, jbh4.jbh4_CONFIGURATION_ID
-- 	, scan.*
-- FROM
-- 	ECAL_RUN_CONFIGURATION_DAT r,
-- 	ECAL_RUN_TYPE_DEF rtd,
-- 	ECAL_RUN_MODE_DEF rmd,
-- 	ECAL_SEQUENCE_DAT s,
-- 	ECAL_SEQUENCE_TYPE_DEF std,
-- 	ECAL_CYCLE_Dat e
-- 	LEFT OUTER join
-- 	ECAL_CCS_CYCLE ccs on  e.cycle_id=ccs.cycle_ID
-- 	left outer join
-- 	ECAL_DCC_CYCLE dcc on  e.cycle_id=dcc.cycle_ID
-- 	left outer join
-- 	ECAL_LASER_CYCLE laser on e.cycle_id=laser.cycle_ID
-- 	left outer join
-- 	ECAL_ltc_CYCLE ltc on  e.cycle_id=ltc.cycle_ID
-- 	left outer join
-- 	ECAL_lts_CYCLE lts on e.cycle_id=lts.cycle_ID
-- 	left outer join
-- 	ECAL_tcc_CYCLE tcc on e.cycle_id=tcc.cycle_ID
-- 	left outer join
-- 	ECAL_ttcci_CYCLE ttcci on  e.cycle_id=ttcci.cycle_ID
-- 	left outer join
-- 	ECAL_matacq_CYCLE matacq on  e.cycle_id=matacq.cycle_ID
-- 	left outer join
-- 	ECAL_jbh4_CYCLE jbh4 on  e.cycle_id=jbh4.cycle_ID
-- 	left outer join
-- 	ECAL_SCAN_cycle scan on e.cycle_id=scan.cycle_id
-- where 
-- 	r.RUN_TYPE_DEF_ID=rtd.DEF_ID
-- 	and r.RUN_MODE_DEF_ID=rmd.DEF_ID
-- 	and r.CONFIG_ID=s.ECAL_CONFIGURATION_ID
-- 	and s.SEQUENCE_TYPE_DEF_ID=std.DEF_ID 
-- 	and e.SEQUENCE_ID=s.SEQUENCE_ID
-- ;
