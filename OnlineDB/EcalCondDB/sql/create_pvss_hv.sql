/* Create tables and indexes for HV data */

CREATE TABLE pvss_hv_imon_dat (
  logic_id	NUMBER(10),
  since		DATE,
  till		DATE,
  imon		NUMBER
);

ALTER TABLE pvss_hv_imon_dat ADD CONSTRAINT pvss_hv_imon_pk PRIMARY KEY (logic_id, since, till);
ALTER TABLE pvss_hv_imon_dat ADD CONSTRAINT pvss_hv_imon_uk UNIQUE (since, till);



CREATE TABLE pvss_hv_i0_dat (
  logic_id	NUMBER(10),
  since		DATE,
  till		DATE,
  i0		NUMBER
);

ALTER TABLE pvss_hv_i0_dat ADD CONSTRAINT pvss_hv_i0_pk PRIMARY KEY (logic_id, since, till);
ALTER TABLE pvss_hv_i0_dat ADD CONSTRAINT pvss_hv_i0_uk UNIQUE (since, till);


 
CREATE TABLE pvss_hv_vmon_dat (
  logic_id	NUMBER(10),
  since		DATE,
  till		DATE,
  vmon		NUMBER
);

ALTER TABLE pvss_hv_vmon_dat ADD CONSTRAINT pvss_hv_vmon_pk PRIMARY KEY (logic_id, since, till);
ALTER TABLE pvss_hv_vmon_dat ADD CONSTRAINT pvss_hv_vmon_uk UNIQUE (since, till);



CREATE TABLE pvss_hv_v0_dat (
  logic_id	NUMBER(10),
  since		DATE,
  till		DATE,
  v0		NUMBER
);

ALTER TABLE pvss_hv_v0_dat ADD CONSTRAINT pvss_hv_v0_pk PRIMARY KEY (logic_id, since, till);
ALTER TABLE pvss_hv_v0_dat ADD CONSTRAINT pvss_hv_v0_uk UNIQUE (since, till);



CREATE TABLE pvss_hv_t_board_dat (
  logic_id	NUMBER(10),
  since		DATE,
  till		DATE,
  t_board	NUMBER
);

ALTER TABLE pvss_hv_t_board_dat ADD CONSTRAINT pvss_hv_t_board_pk PRIMARY KEY (logic_id, since, till);
ALTER TABLE pvss_hv_t_board_dat ADD CONSTRAINT pvss_hv_t_board_uk UNIQUE (since, till);



/* Function for converting PVSS ID to logic_id */
@pvss_hvchannellogicid_func
@pvss_hvboardlogicid_func


/* Triggers for creating seamless IOVs */
CREATE OR REPLACE
TRIGGER pvss_hv_imon_tg
  BEFORE INSERT ON pvss_hv_imon_dat
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_date('pvss_hv_imon_dat', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/

CREATE OR REPLACE
TRIGGER pvss_hv_i0_tg
  BEFORE INSERT ON pvss_hv_i0_dat
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_date('pvss_hv_i0_dat', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/

CREATE OR REPLACE
TRIGGER pvss_hv_vmon_tg
  BEFORE INSERT ON pvss_hv_vmon_dat
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_date('pvss_hv_vmon_dat', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/

CREATE OR REPLACE
TRIGGER pvss_hv_v0_tg
  BEFORE INSERT ON pvss_hv_v0_dat
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_date('pvss_hv_v0_dat', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/

CREATE OR REPLACE
TRIGGER pvss_hv_t_board_tg
  BEFORE INSERT ON pvss_hv_t_board_dat
  REFERENCING NEW AS newiov
  FOR EACH ROW
  
begin
  update_online_pvss_iov_date('pvss_hv_t_board_dat', :newiov.since, :newiov.till,:newiov.logic_id);
end;
/
