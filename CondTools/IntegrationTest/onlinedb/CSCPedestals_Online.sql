REM
REM Table PEDESTALS
REM
create table PEDESTALS (
record_id         number NOT NULL,
run_num           number UNIQUE NOT NULL,
data_taking_time  date DEFAULT sysdate NOT NULL,
insertion_time    date DEFAULT sysdate NOT NULL);
REM
REM Adding constraints for table PEDESTALS
REM
alter table PEDESTALS
  add constraint ped_run_pk primary key (record_id);

REM
REM Table PEDESTASLS_MAP
REM
create table PEDESTALS_MAP (
map_id      number NOT NULL,
record_id   number NOT NULL,
map_index   number NOT NULL,
layer_id    number NOT NULL);
REM
REM Adding constraints for table PEDESTALS_MAP
REM
alter table PEDESTALS_MAP add (
   constraint ped_map_pk primary key (map_id),
   unique (record_id,map_index),
   unique (record_id,layer_id),
   constraint ped_map_fk foreign key (record_id)
                         references pedestals(record_id));

REM
REM Table PEDESTALS_DATA
REM
create table PEDESTALS_DATA (
map_id     number NOT NULL,
vec_index  number(5) NOT NULL,
ped        binary_float NOT NULL,
rms        binary_float NOT NULL);
REM
REM Adding constraints for table PEDESTALS_DATA
REM
alter table PEDESTALS_DATA add (
   constraint ped_data_pk primary key (map_id,vec_index),
   constraint ped_data_fk foreign key (map_id)
                          references pedestals_map(map_id));
