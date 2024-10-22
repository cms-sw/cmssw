-- SQL script used to add the double weights key in the ConfDB

alter table fe_config_main drop constraint fe_config_main_pk ;

ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_L2_UNIQUE_uk ;

ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_MAIN_to_PED_fk ;
ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_MAIN_to_lin_fk ;
ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_MAIN_to_lut_fk ;
ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_MAIN_to_fgr_fk ;
ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_MAIN_to_sli_fk ;
ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_MAIN_to_WEIGHT_fk ;
ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_MAIN_to_spi_fk ;
ALTER TABLE FE_CONFIG_MAIN drop CONSTRAINT FE_CONFIG_MAIN_to_tim_fk ;


alter table fe_config_main rename to old_fe_config_main3;


CREATE TABLE FE_CONFIG_MAIN (
conf_id NUMBER NOT NULL,                                                               
ped_conf_id NUMBER NOT NULL,                                                          
lin_conf_id NUMBER NOT NULL,                                                       
lut_conf_id NUMBER NOT NULL,                                                      
fgr_conf_id NUMBER NOT NULL,                                                
sli_conf_id NUMBER NOT NULL,                                            
wei_conf_id NUMBER NOT NULL,                                      
spi_conf_id NUMBER DEFAULT 0 NOT NULL, 
tim_conf_id NUMBER DEFAULT 0 NOT NULL,                                         
bxt_conf_id NUMBER NOT NULL,                                      
btt_conf_id NUMBER NOT NULL,                                      
bst_conf_id NUMBER DEFAULT 0 NOT NULL,                                   
tag         VARCHAR2(100),                                                 
version NUMBER  NOT NULL,                                                     
description VARCHAR2(200)  ,                                                                    
db_timestamp            TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
coke_conf_id NUMBER DEFAULT 0 NOT NULL
wei2_conf_id NUMBER DEFAULT 1 NOT NULL
);


insert into fe_config_main (conf_id, ped_conf_id ,lin_conf_id, 
lut_conf_id, fgr_conf_id, sli_conf_id, wei_conf_id, spi_conf_id, tim_conf_id, bxt_conf_id, 
btt_conf_id, bst_conf_id, tag, version, description, db_timestamp,coke_conf_id ) (select conf_id, ped_conf_id ,lin_conf_id, 
lut_conf_id, fgr_conf_id, sli_conf_id, wei_conf_id, spi_conf_id, tim_conf_id, bxt_conf_id, 
btt_conf_id, bst_conf_id, tag, version, description, db_timestamp,coke_conf_id from old_fe_config_main3 ) ;

-- drop table old_fe_config_main3;


ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_PK PRIMARY KEY (CONF_ID);

ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_L2_UNIQUE_uk UNIQUE (tag,version);

ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_PED_fk FOREIGN KEY (ped_conf_id) REFERENCES FE_CONFIG_PED_INFO (ped_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_lin_fk FOREIGN KEY (lin_conf_id) REFERENCES FE_CONFIG_LIN_INFO (lin_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_lut_fk FOREIGN KEY (lut_conf_id) REFERENCES FE_CONFIG_LUT_INFO (lut_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_fgr_fk FOREIGN KEY (fgr_conf_id) REFERENCES FE_CONFIG_fgr_INFO (fgr_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_sli_fk FOREIGN KEY (sli_conf_id) REFERENCES FE_CONFIG_sliding_INFO (sli_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_WEIGHT_fk FOREIGN KEY (wei_conf_id) REFERENCES FE_CONFIG_WEIGHT_INFO (wei_conf_id);

ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_spi_fk FOREIGN KEY (spi_conf_id) REFERENCES FE_CONFIG_spike_INFO (spi_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_tim_fk FOREIGN KEY (tim_conf_id) REFERENCES FE_CONFIG_time_INFO (tim_conf_id);
--  ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_cok_fk FOREIGN KEY (coke_conf_id) REFERENCES FE_CONFIG_coke_INFO (coke_conf_id);


CREATE OR REPLACE TRIGGER fe_config_main_auto3_ver_tg
  BEFORE INSERT ON FE_CONFIG_MAIN
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_MAIN', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;



CREATE TABLE FE_CONFIG_WEIGHT2_INFO (
 wei2_conf_id NUMBER(10) NOT NULL,
 number_of_groups NUMBER(10) , -- (the number of groups of weights)
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL,
 TAG VARCHAR2(100)
);
ALTER TABLE FE_CONFIG_WEIGHT2_INFO ADD CONSTRAINT  FE_CONFIG_WEIGHT2_INFO_PK PRIMARY KEY (wei2_conf_id);

insert into FE_CONFIG_WEIGHT2_INFO (wei2_conf_id, number_of_groups, TAG) values (1,0, 'NoOddWeights' ) ;


create table FE_WEIGHT2_PER_GROUP_DAT(
 wei2_conf_id number not null,
 group_id number(10) not null,
 W0 NUMBER,
 W1 NUMBER,
 W2 NUMBER,
 W3 NUMBER,
 W4 NUMBER,
 W5 NUMBER
 );

ALTER TABLE FE_WEIGHT2_PER_GROUP_DAT ADD CONSTRAINT FE_WEIGHT2_PER_GROUP_pk PRIMARY KEY (wei2_conf_id , group_id);
ALTER TABLE FE_WEIGHT2_PER_GROUP_DAT ADD CONSTRAINT FE_WEIGHT2_PER_GROUP_fk foreign KEY (wei2_conf_id) REFERENCES FE_CONFIG_WEIGHT2_INFO (wei2_conf_id);

insert into FE_CONFIG_WEIGHT2_PER_GROUP (1,0,0,0,0,0,0,0);
insert into FE_CONFIG_WEIGHT2_PER_GROUP (1,1,0,0,0,0,0,0);


CREATE TABLE FE_CONFIG_WEIGHT2_DAT (
 wei2_conf_id NUMBER NOT NULL,
 logic_id NUMBER(10) not null, -- ( of the strip)
 group_id number(10) not null);

ALTER TABLE FE_CONFIG_WEIGHT2_DAT ADD CONSTRAINT FE_CONFIG_WEIGHT2_fk  FOREIGN KEY (wei2_conf_id) REFERENCES FE_CONFIG_WEIGHT2_INFO (wei2_conf_id);
ALTER TABLE FE_CONFIG_WEIGHT2_DAT ADD CONSTRAINT FE_CONFIG_WEIGHT2_fk2  FOREIGN KEY (wei2_conf_id, group_id) 
REFERENCES FE_WEIGHT2_PER_GROUP_DAT (wei2_conf_id, group_id);

insert into FE_CONFIG_WEIGHT2_DAT (wei2_conf_id,logic_id,group_id) (select 1,logic_id,group_id from FE_CONFIG_WEIGHT_DAT where wei_conf_id=480);


create table FE_WEIGHT2_MODE_DAT(
 wei2_conf_id number not null,
 EnableEBOddFilter number DEFAULT 0 NOT NULL,
 EnableEEOddFilter number DEFAULT 0 NOT NULL,
 EnableEBOddPeakFinder number DEFAULT 0 NOT NULL,
 EnableEEOddPeakFinder number DEFAULT 0 NOT NULL,
 DisableEBEvenPeakFinder number DEFAULT 0 NOT NULL,
 FenixEBStripOutput number DEFAULT 0 NOT NULL,
 FenixEEStripOutput number DEFAULT 0 NOT NULL,
 FenixEBStripInfobit2 number DEFAULT 0 NOT NULL,
 FenixEEStripInfobit2 number DEFAULT 0 NOT NULL,
 EBFenixTcpOutput number DEFAULT 0 NOT NULL,
 EBFenixTcpInfobit1 number DEFAULT 0 NOT NULL,
 FenixPar12  number DEFAULT 0 NOT NULL,
 FenixPar13  number DEFAULT 0 NOT NULL,
 FenixPar14  number DEFAULT 0 NOT NULL,
 FenixPar15  number DEFAULT 0 NOT NULL
 );

ALTER TABLE FE_WEIGHT2_MODE_DAT ADD CONSTRAINT FE_WEIGHT2_MODE_pk PRIMARY KEY (wei2_conf_id);
ALTER TABLE FE_WEIGHT2_MODE_DAT ADD CONSTRAINT FE_WEIGHT2_MODE_fk foreign KEY (wei2_conf_id) REFERENCES FE_CONFIG_WEIGHT2_INFO (wei2_conf_id);

INSERT into FE_WEIGHT2_MODE_DAT(wei2_conf_id) values (1); 



CREATE SEQUENCE FE_CONFIG_WEIGHT2GROUP_SQ INCREMENT BY 1 START WITH 1;
select FE_CONFIG_WEIGHT2GROUP_SQ.NextVal from DUAL; 

-- this selects the first number that we have already used

/*
 *  the id is auto-incremented at each time you insert in the table 
 *  no need to bother about inserting conf_id
 */

CREATE trigger FE_CONFIG_WEI2_TRG
before insert on FE_CONFIG_WEIGHT2_INFO
for each row
begin
select FE_CONFIG_WEIGHT2GROUP_SQ.NextVal into :new.wei2_conf_id from dual;
end;
/





