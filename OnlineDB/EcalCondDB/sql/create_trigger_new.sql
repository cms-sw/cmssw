/*
 *  Creates all the data tables required to configure the trigger and FE for the trigger
 */


CREATE TABLE FE_CONFIG_MAIN (
conf_id NUMBER NOT NULL, -- (the most important trigger key)
ped_conf_id NUMBER NOT NULL, -- (the link to the pedestals)
lin_conf_id NUMBER NOT NULL, -- (the link to the lin table)
lut_conf_id NUMBER NOT NULL, -- (the link to the LUT table)
fgr_conf_id NUMBER NOT NULL, -- (the link to the fine grain table)
sli_conf_id NUMBER NOT NULL, -- (the link to the sliding window table)
wei_conf_id NUMBER NOT NULL, -- (the link to the weight configuration table)
spi_conf_id NUMBER NOT NULL, -- (the link to the spike killer conf table) 
bxt_conf_id NUMBER NOT NULL, -- (the link to the bad xt configuration table)
btt_conf_id NUMBER NOT NULL, -- (the link to the bad tt configuration table)
bst_conf_id NUMBER NOT NULL, -- (the link to the bad strip configuration table)
tag         VARCHAR2(100), -- (a comment if you want to add it)
version NUMBER  NOT NULL, -- (the most important trigger key)
description VARCHAR2(200)  , -- (just a string )
db_timestamp		TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);

ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_PK PRIMARY KEY (CONF_ID);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_L1_UNIQUE_uk UNIQUE (ped_conf_id,lin_conf_id,lut_conf_id,fgr_conf_id,sli_conf_id,wei_conf_id, spi_conf_id, bxt_conf_id, btt_conf_id, bst_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_L2_UNIQUE_uk UNIQUE (tag,version);

CREATE SEQUENCE FE_CONFIG_MAIN_SQ INCREMENT BY 1 START WITH 1 nocache;



CREATE OR REPLACE TRIGGER fe_config_main_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_MAIN
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_MAIN', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;




CREATE TABLE FE_CONFIG_PED_INFO (
 ped_conf_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100), 
 version number, 	
 iov_id NUMBER(10) , -- references the condition DB table used 
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_PED_INFO ADD CONSTRAINT  FE_CONFIG_PED_INFO_PK PRIMARY KEY (ped_conf_id);
ALTER TABLE FE_CONFIG_PED_INFO ADD CONSTRAINT  FE_CONFIG_PED_UNIQUE_uk UNIQUE (tag,version);


CREATE TABLE FE_CONFIG_LIN_INFO (
 lin_conf_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100),
 version number	,
 iov_id NUMBER(10) , -- references the condition DB table used 
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_lin_INFO ADD CONSTRAINT  FE_CONFIG_lin_INFO_PK PRIMARY KEY (lin_conf_id);
ALTER TABLE FE_CONFIG_lin_INFO ADD CONSTRAINT  FE_CONFIG_lin_UNIQUE_uk UNIQUE (tag,version);



CREATE TABLE FE_CONFIG_LUT_INFO (
 lut_conf_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100),
 version number	,
 number_of_groups NUMBER(10) , 
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_lut_INFO ADD CONSTRAINT  FE_CONFIG_lut_INFO_PK PRIMARY KEY (lut_conf_id);
ALTER TABLE FE_CONFIG_lut_INFO ADD CONSTRAINT  FE_CONFIG_lut_UNIQUE_uk UNIQUE (tag,version);



CREATE TABLE FE_CONFIG_fgr_INFO (
 fgr_conf_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100), 
 version number	,
 number_of_groups NUMBER(10) , 
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_fgr_INFO ADD CONSTRAINT  FE_CONFIG_fgr_INFO_PK PRIMARY KEY (fgr_conf_id);
ALTER TABLE FE_CONFIG_fgr_INFO ADD CONSTRAINT  FE_CONFIG_fgr_UNIQUE_uk UNIQUE (tag,version);



CREATE TABLE FE_CONFIG_sliding_INFO (
 sli_conf_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100), 
 version number, 
 iov_id NUMBER(10) , -- references the condition DB table used 
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_sliding_INFO ADD CONSTRAINT  FE_CONFIG_SLIDING_INFO_PK PRIMARY KEY (sli_conf_id);
ALTER TABLE FE_CONFIG_sliding_INFO ADD CONSTRAINT  FE_CONFIG_sliding_UNIQUE_uk UNIQUE (tag,version);

CREATE TABLE FE_CONFIG_spike_INFO (
 spi_conf_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100),
 version number,
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL
);
ALTER TABLE FE_CONFIG_spike_INFO ADD CONSTRAINT  FE_CONFIG_Spike_INFO_PK PRIMARY KEY (spi_conf_id);
ALTER TABLE FE_CONFIG_spike_INFO ADD CONSTRAINT  FE_CONFIG_spike_UNIQUE_uk UNIQUE (tag,version);



CREATE TABLE FE_CONFIG_WEIGHT_INFO (
 wei_conf_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100), 
 version number, 	
 number_of_groups NUMBER(10) , -- (the number of groups of weights)
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_WEIGHT_INFO ADD CONSTRAINT  FE_CONFIG_WEIGHT_INFO_PK PRIMARY KEY (wei_conf_id);
ALTER TABLE FE_CONFIG_weight_INFO ADD CONSTRAINT  FE_CONFIG_weight_UNIQUE_uk UNIQUE (tag,version);


CREATE TABLE FE_CONFIG_BadCrystals_INFO (
 rec_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100),
 version number	,
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_BadCrystals_INFO ADD CONSTRAINT  FE_CONFIG_Badxt_INFO_PK PRIMARY KEY (rec_id);
ALTER TABLE FE_CONFIG_badcrystals_INFO ADD CONSTRAINT  FE_CONFIG_badxt_UNIQUE_uk UNIQUE (tag,version);


CREATE TABLE FE_CONFIG_BadTT_INFO (
 rec_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100),
 version number	,
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_BadTT_INFO ADD CONSTRAINT  FE_CONFIG_BadTT_INFO_PK PRIMARY KEY (rec_id);
ALTER TABLE FE_CONFIG_badtt_INFO ADD CONSTRAINT  FE_CONFIG_badtt_UNIQUE_uk UNIQUE (tag,version);

CREATE TABLE FE_CONFIG_BadST_INFO (
 rec_id NUMBER(10) NOT NULL,
 TAG VARCHAR2(100),
 version number	,
 db_timestamp  TIMESTAMP DEFAULT SYSTIMESTAMP NOT NULL	
);
ALTER TABLE FE_CONFIG_BadST_INFO ADD CONSTRAINT  FE_CONFIG_BadST_INFO_PK PRIMARY KEY (rec_id);
ALTER TABLE FE_CONFIG_badSt_INFO ADD CONSTRAINT  FE_CONFIG_badSt_UNIQUE_uk UNIQUE (tag,version);




CREATE OR REPLACE TRIGGER fe_config_ped_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_ped_info
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_PED_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;


CREATE OR REPLACE TRIGGER fe_config_lin_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_lin_info
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_LIN_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;


CREATE OR REPLACE TRIGGER fe_config_lut_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_LUT_INFO
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_LUT_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;


CREATE OR REPLACE TRIGGER fe_config_FGR_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_FGR_INFO
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_FGR_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;


CREATE OR REPLACE TRIGGER fe_config_WEI_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_WEIGHT_INFO
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_WEIGHT_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;


CREATE OR REPLACE TRIGGER fe_config_sli_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_sliding_info
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_SLIDING_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;

CREATE OR REPLACE TRIGGER fe_config_spi_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_spike_info
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_Spike_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;


CREATE OR REPLACE TRIGGER fe_config_bxt_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_Badcrystals_INFO
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_BADCRYSTALS_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;

CREATE OR REPLACE TRIGGER fe_config_btt_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_BADTT_INFO
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_BADTT_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;

CREATE OR REPLACE TRIGGER fe_config_bst_info_auto_ver_tg
  BEFORE INSERT ON FE_CONFIG_BADST_INFO
  FOR EACH ROW
    begin
  select test_update_tag_and_version('FE_CONFIG_BADST_INFO', :new.tag, :new.version) into :new.version from dual;
end;
/
SHOW ERRORS;









/*
 *  here we have the pedestals - 3 values per crystal (1 per gain)
 */


CREATE TABLE FE_CONFIG_PED_DAT (
 ped_conf_id NUMBER(10) NOT NULL,
 logic_id NUMBER(10) not null, 
 mean_12 NUMBER(10),
 mean_6 NUMBER(10),
 mean_1 NUMBER(10));

ALTER TABLE FE_CONFIG_PED_DAT ADD CONSTRAINT FE_CONFIG_PED_pk PRIMARY KEY (ped_conf_id, logic_id);
ALTER TABLE FE_CONFIG_PED_DAT ADD CONSTRAINT FE_CONFIG_PED_fk FOREIGN KEY (ped_conf_id) REFERENCES FE_CONFIG_PED_INFO (ped_conf_id);



/*
 * adc_to_gev is the LSB equivalent energy typical is 0.035 GeV/ADC 
 *  here we have the absolute calibration - 1 value per barrel + 1 value per EC maybe
 * logic_id=2000000001 for EE and 1000000000 for EB
 * Energy_crystal=FE_CONFIG_calib_lsb_DAT.adc_to_gev * fe_config_calib_dat.calibration * Peak(ADCCounts)
 * and at higher gains 
 * Energy_crystal= FE_CONFIG_calib_lsb_DAT.adc_to_gev * fe_config_calib_dat.calibration * gain_ratio * Peak(ADCCounts) 
 */


/*
 *  here we have the linearization - 1 value per crystal
 */

CREATE TABLE FE_CONFIG_lin_DAT (
  lin_conf_id        NUMBER(10),
  logic_id              NUMBER(10), -- (crystal)
  multx12                Number,
  multx6                 number,
  multx1                 number,
  shift12                number,
  shift6                 number,
  shift1                 number
);

ALTER TABLE FE_CONFIG_lin_DAT ADD CONSTRAINT FE_CONFIG_lin_pk PRIMARY KEY (lin_conf_id, logic_id);
ALTER TABLE FE_CONFIG_lin_DAT ADD CONSTRAINT FE_CONFIG_lin_fk FOREIGN KEY (lin_conf_id) REFERENCES FE_CONFIG_lin_INFO (lin_conf_id);


/*
 *  here we have the linearization parameters used to compute the lin coeff.
 */





/*
 *  here we have the weights and the sliding window parameters - values per strip
 */


CREATE TABLE FE_CONFIG_sliding_DAT (
  sli_conf_id        NUMBER(10),
  logic_id              NUMBER(10), -- (strip)
  sliding                 NUMBER(10)
);

ALTER TABLE FE_CONFIG_sliding_DAT ADD CONSTRAINT FE_CONFIG_sliding_pk PRIMARY KEY (sli_conf_id, logic_id);
ALTER TABLE FE_CONFIG_sliding_DAT ADD CONSTRAINT FE_CONFIG_sliding_fk FOREIGN KEY (sli_conf_id) REFERENCES FE_CONFIG_sliding_INFO (sli_conf_id);

CREATE TABLE FE_CONFIG_spike_DAT (
  spi_conf_id        NUMBER(10),
  logic_id              NUMBER(10), -- (barrel tower)                                                                                
  spike_threshold       NUMBER(10)
);

ALTER TABLE FE_CONFIG_spike_dat ADD CONSTRAINT FE_CONFIG_spike_pk PRIMARY KEY (spi_conf_id, logic_id);
ALTER TABLE FE_CONFIG_spike_DAT ADD CONSTRAINT FE_CONFIG_spike_fk FOREIGN KEY (spi_conf_id) REFERENCES FE_CONFIG_spike_INFO (spi_conf_id);




create table FE_WEIGHT_PER_GROUP_DAT(
 wei_conf_id number not null,
 group_id number(10) not null,
 W0 NUMBER,
 W1 NUMBER,
 W2 NUMBER,
 W3 NUMBER,
 W4 NUMBER	
 );

ALTER TABLE FE_WEIGHT_PER_GROUP_DAT ADD CONSTRAINT FE_WEIGHT_PER_GROUP_pk PRIMARY KEY (wei_conf_id , group_id);
ALTER TABLE FE_WEIGHT_PER_GROUP_DAT ADD CONSTRAINT FE_WEIGHT_PER_GROUP_fk foreign KEY (wei_conf_id) REFERENCES FE_CONFIG_WEIGHT_INFO (wei_conf_id);


CREATE TABLE FE_CONFIG_WEIGHT_DAT (
 wei_conf_id NUMBER NOT NULL,
 logic_id NUMBER(10) not null, -- ( of the strip)
 group_id number(10) not null);

ALTER TABLE FE_CONFIG_WEIGHT_DAT ADD CONSTRAINT FE_CONFIG_WEIGHT_fk  FOREIGN KEY (wei_conf_id) REFERENCES FE_CONFIG_WEIGHT_INFO (wei_conf_id);
ALTER TABLE FE_CONFIG_WEIGHT_DAT ADD CONSTRAINT FE_CONFIG_WEIGHT_fk2  FOREIGN KEY (wei_conf_id, group_id) REFERENCES FE_WEIGHT_PER_GROUP_DAT (wei_conf_id, group_id);


/*
 *  here we have the LUT and fine grain para - values per TT
 */



create table FE_LUT_PER_GROUP_DAT(
 lut_conf_id number(10) not null,
 group_id number(10) not null,
 lut_id NUMBER(10),
 lut_value NUMBER
 );

ALTER TABLE FE_LUT_PER_GROUP_DAT ADD CONSTRAINT FE_LUT_PER_GROUP_pk PRIMARY KEY (lut_conf_id, group_id , lut_id);
ALTER TABLE FE_LUT_PER_GROUP_DAT ADD CONSTRAINT FE_LUT_PER_GROUP_fk foreign KEY (lut_conf_id) REFERENCES FE_CONFIG_LUT_INFO (lut_conf_id);


CREATE TABLE FE_CONFIG_LUT_DAT (
 lut_conf_id NUMBER NOT NULL,
 logic_id NUMBER(10) not null, -- ( of the TT)
 group_id number(10) not null);

ALTER TABLE FE_CONFIG_LUT_DAT ADD CONSTRAINT FE_CONFIG_LUT_fk  FOREIGN KEY (lut_conf_id) REFERENCES FE_CONFIG_LUT_INFO (lut_conf_id);



create table FE_fgr_PER_GROUP_DAT(
 fgr_conf_id NUMBER NOT NULL,
 group_id number(10) not null,
 threshold_low NUMBER,
 threshold_high NUMBER,
 ratio_low NUMBER,
 ratio_high NUMBER,
 lut_value NUMBER(10)	
 );

ALTER TABLE FE_fgr_PER_GROUP_DAT ADD CONSTRAINT FE_fgr_PER_GROUP_pk PRIMARY KEY (fgr_conf_id, group_id);


CREATE TABLE FE_CONFIG_FGR_DAT (
 fgr_conf_id NUMBER NOT NULL,
 logic_id NUMBER(10) not null, -- ( of the TT)
 group_id number(10) not null);

ALTER TABLE FE_CONFIG_FGR_DAT ADD CONSTRAINT FE_CONFIG_FGR_pk PRIMARY KEY (fgr_conf_id, logic_id);
ALTER TABLE FE_CONFIG_FGR_DAT ADD CONSTRAINT FE_CONFIG_FGR_fk  FOREIGN KEY (fgr_conf_id) REFERENCES FE_CONFIG_FGR_INFO (fgr_conf_id);

/* endcap part not by groups */
CREATE TABLE FE_CONFIG_FGREEST_DAT (
 fgr_conf_id        NUMBER(10),
 logic_id              NUMBER(10), -- (tower by tcc and tt)
 threshold                Number(10),
 lut_fg                 number(10)
);

ALTER TABLE FE_CONFIG_FGREEST_DAT ADD CONSTRAINT FE_CONFIG_FGREEST_pk PRIMARY KEY (fgr_conf_id, logic_id);
ALTER TABLE FE_CONFIG_FGREEST_DAT ADD CONSTRAINT FE_CONFIG_FGREEST_fk FOREIGN KEY (fgr_conf_id) REFERENCES FE_CONFIG_FGR_INFO (fgr_conf_id);

CREATE TABLE FE_CONFIG_FGREETT_DAT (
 fgr_conf_id        NUMBER(10),
 logic_id              NUMBER(10), -- (strip)
 lut_value                 number(10)
);

ALTER TABLE FE_CONFIG_FGREETT_DAT ADD CONSTRAINT FE_CONFIG_FGREETT_pk PRIMARY KEY (fgr_conf_id, logic_id);
ALTER TABLE FE_CONFIG_FGREETT_DAT ADD CONSTRAINT FE_CONFIG_FGREETT_fk FOREIGN KEY (fgr_conf_id) REFERENCES FE_CONFIG_FGR_INFO (fgr_conf_id);



CREATE TABLE FE_CONFIG_BadCrystals_DAT (
rec_id NUMBER(10) NOT NULL,
tcc_id  NUMBER(10),
fed_id NUMBER(10),
tt_id  NUMBER(10),
cry_id NUMBER(10),
 status NUMBER(10));

ALTER TABLE FE_CONFIG_BadCrystals_DAT ADD CONSTRAINT FE_CONFIG_BXT_pk PRIMARY KEY (rec_id,tcc_id,fed_id,tt_id,cry_id);
 ALTER TABLE FE_CONFIG_BadCrystals_DAT ADD CONSTRAINT FE_CONFIG_BXT_fk FOREIGN KEY (rec_id) REFERENCES FE_CONFIG_BadCrystals_INFO (rec_id); 
/* ALTER TABLE FE_CONFIG_BadCrystals_DAT ADD CONSTRAINT FE_CONFIG_BXT_fk FOREIGN KEY  (REC_ID) REFERENCES COND2CONF_INFO (REC_ID); */

CREATE TABLE FE_CONFIG_BadTT_DAT (
rec_id NUMBER(10) NOT NULL,
tcc_id  NUMBER(10),
fed_id NUMBER(10),
tt_id  NUMBER(10),
 status NUMBER(10));

ALTER TABLE FE_CONFIG_BadTT_DAT ADD CONSTRAINT FE_CONFIG_BTT_pk PRIMARY KEY (rec_id,tcc_id,fed_id,tt_id );
ALTER TABLE FE_CONFIG_BadTT_DAT ADD CONSTRAINT FE_CONFIG_BTT_fk FOREIGN KEY (rec_id) REFERENCES FE_CONFIG_BadTT_INFO (rec_id); 
/* ALTER TABLE FE_CONFIG_BadTT_DAT ADD CONSTRAINT FE_CONFIG_BTT_fk FOREIGN KEY (rec_id) REFERENCES COND2CONF_INFO (rec_id); */

CREATE TABLE FE_CONFIG_BadST_DAT (
rec_id NUMBER(10) NOT NULL,
tcc_id  NUMBER(10),
fed_id NUMBER(10),
tt_id  NUMBER(10),
st_is  NUMBER(2),
status NUMBER(10));

ALTER TABLE FE_CONFIG_BadST_DAT ADD CONSTRAINT FE_CONFIG_BST_pk PRIMARY KEY (rec_id,tcc_id,fed_id,tt_id,st_id );
ALTER TABLE FE_CONFIG_BadST_DAT ADD CONSTRAINT FE_CONFIG_BST_fk FOREIGN KEY (rec_id) REFERENCES FE_CONFIG_BadST_INFO (rec_id); 



/* now the main table constraints */


ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_PED_fk FOREIGN KEY (ped_conf_id) REFERENCES FE_CONFIG_PED_INFO (ped_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_lin_fk FOREIGN KEY (lin_conf_id) REFERENCES FE_CONFIG_LIN_INFO (lin_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_lut_fk FOREIGN KEY (lut_conf_id) REFERENCES FE_CONFIG_LUT_INFO (lut_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_fgr_fk FOREIGN KEY (fgr_conf_id) REFERENCES FE_CONFIG_fgr_INFO (fgr_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_sli_fk FOREIGN KEY (sli_conf_id) REFERENCES FE_CONFIG_sliding_INFO (sli_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_spi_fk FOREIGN KEY (spi_conf_id) REFERENCES FE_CONFIG_spike_INFO (spi_conf_id);
ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_WEIGHT_fk FOREIGN KEY (wei_conf_id) REFERENCES FE_CONFIG_WEIGHT_INFO (wei_conf_id);
/*  ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_BXT_fk FOREIGN KEY (bxt_conf_id) REFERENCES FE_CONFIG_BadCrystals_INFO (bxt_conf_id); */
/* ALTER TABLE FE_CONFIG_MAIN ADD CONSTRAINT FE_CONFIG_MAIN_to_BTT_fk FOREIGN KEY (btt_conf_id) REFERENCES FE_CONFIG_BadTT_INFO (btt_conf_id); */


CREATE TABLE FE_CONFIG_linparam_DAT (
  lin_conf_id        NUMBER(10),
  logic_id           NUMBER(10), -- (crystal)
  etsat                Number
);

ALTER TABLE FE_CONFIG_linparam_DAT ADD CONSTRAINT FE_CONFIG_linparam_pk PRIMARY KEY (lin_conf_id, logic_id);
ALTER TABLE FE_CONFIG_linparam_DAT ADD CONSTRAINT FE_CONFIG_linparam_fk FOREIGN KEY (lin_conf_id) REFERENCES FE_CONFIG_lin_INFO (lin_conf_id);


CREATE TABLE FE_CONFIG_lutparam_DAT (
  lut_conf_id        NUMBER(10),
  logic_id           NUMBER(10), -- (crystal)
  etsat                Number,
  ttthreshlow         Number,
  ttthreshhigh         Number
);

ALTER TABLE FE_CONFIG_lutparam_DAT ADD CONSTRAINT FE_CONFIG_lutparam_pk PRIMARY KEY (lut_conf_id, logic_id);
ALTER TABLE FE_CONFIG_lutparam_DAT ADD CONSTRAINT FE_CONFIG_lutparam_fk FOREIGN KEY (lut_conf_id) REFERENCES FE_CONFIG_lut_INFO (lut_conf_id);

CREATE TABLE FE_CONFIG_fgrparam_DAT (
  fgr_conf_id        NUMBER(10),
  logic_id           NUMBER(10), -- (crystal)
  fg_lowthresh         Number,
  fg_highthresh         Number,
  fg_lowratio         Number,
  fg_highratio         Number
);

ALTER TABLE FE_CONFIG_fgrparam_DAT ADD CONSTRAINT FE_CONFIG_fgrparam_pk PRIMARY KEY (fgr_conf_id, logic_id);
ALTER TABLE FE_CONFIG_fgrparam_DAT ADD CONSTRAINT FE_CONFIG_fgrparam_fk FOREIGN KEY (fgr_conf_id) REFERENCES FE_CONFIG_fgr_INFO (fgr_conf_id);





CREATE SEQUENCE FE_CONFIG_PED_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_LIN_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_LUT_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_FGR_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_SLI_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_Spi_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_WEIGHT_SQ INCREMENT BY 1 START WITH  1 nocache;
CREATE SEQUENCE FE_CONFIG_LUTGROUP_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_FGRGROUP_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_WEIGHTGROUP_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_BXT_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_BTT_SQ INCREMENT BY 1 START WITH 1 nocache;
CREATE SEQUENCE FE_CONFIG_BST_SQ INCREMENT BY 1 START WITH 1 nocache;



/* create synonym CHANNELVIEW              for cms_ecal_cond.CHANNELVIEW              ; */
/* create synonym VIEWDESCRIPTION          for cms_ecal_cond.VIEWDESCRIPTION          ; */






