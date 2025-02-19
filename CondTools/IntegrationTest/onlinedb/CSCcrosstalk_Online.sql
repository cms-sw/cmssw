REM
REM Table CROSSTALK
REM
create table CROSSTALK (
record_id         number NOT NULL,
run_num           number UNIQUE NOT NULL,
data_taking_time  date DEFAULT sysdate NOT NULL,
insertion_time    date DEFAULT sysdate NOT NULL);
REM
REM Adding constraints for table CROSSTALK
REM
alter table CROSSTALK
  add constraint xtalk_run_pk primary key (record_id);

REM
REM Table CROSSTALK_MAP
REM
create table CROSSTALK_MAP (
map_id      number NOT NULL,
record_id   number NOT NULL,
map_index   number NOT NULL,
layer_id    number NOT NULL);
REM
REM Adding constraints for table CROSSTALK_MAP
REM
alter table CROSSTALK_MAP add (
   constraint xtalk_map_pk primary key (map_id),
   unique (record_id,map_index),
   unique (record_id,layer_id),
   constraint xtalk_map_fk foreign key (record_id)
                          references crosstalk(record_id));

REM
REM Table CROSSTALK_DATA
REM
create table CROSSTALK_DATA (
map_id         number NOT NULL,
vec_index      number(5) NOT NULL,
xtalk_slope_right     number(15,10) NOT NULL,
xtalk_intercept_right number(15,9) NOT NULL,
xtalk_chi2_right      number(15,6) NOT NULL,
xtalk_slope_left      number(15,10) NOT NULL,
xtalk_intercept_left  number(15,9) NOT NULL,
xtalk_chi2_left       number(15,6) NOT NULL);
REM
REM Adding constraints for table CROSSTALK_DATA
REM
alter table CROSSTALK_DATA add (
   constraint xtalk_data_pk primary key (map_id,vec_index),
   constraint xtalk_data_fk foreign key (map_id)
                          references crosstalk_map(map_id));
