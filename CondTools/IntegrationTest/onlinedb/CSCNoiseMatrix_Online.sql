REM
REM Table NOISEMATRIX
REM
create table NOISEMATRIX (
record_id         number NOT NULL,
run_num           number UNIQUE NOT NULL,
data_taking_time  date DEFAULT sysdate NOT NULL,
insertion_time    date DEFAULT sysdate NOT NULL);
REM
REM Adding constraints for table NOISEMATRIX
REM
alter table NOISEMATRIX
  add constraint noisematrix_run_pk primary key (record_id);

REM
REM Table NOISEMATRIX_MAP
REM
create table NOISEMATRIX_MAP (
map_id      number NOT NULL,
record_id   number NOT NULL,
map_index   number NOT NULL,
layer_id    number NOT NULL);
REM
REM Adding constraints for table NOISEMATRIX_MAP
REM
alter table NOISEMATRIX_MAP add (
   constraint noisematrix_map_pk primary key (map_id),
   unique (record_id,map_index),
   unique (record_id,layer_id),
   constraint noisematrix_map_fk foreign key (record_id)
                          references noisematrix(record_id));

REM
REM Table NOISEMATRIX_DATA
REM
create table NOISEMATRIX_DATA (
map_id      number NOT NULL,
vec_index   number(5) NOT NULL,
elem33      number(15,6) NOT NULL,
elem34      number(15,6) NOT NULL,
elem35      number(15,6) NOT NULL,
elem44      number(15,6) NOT NULL,
elem45      number(15,6) NOT NULL,
elem46      number(15,6) NOT NULL,
elem55      number(15,6) NOT NULL,
elem56      number(15,6) NOT NULL,
elem57      number(15,6) NOT NULL,
elem66      number(15,6) NOT NULL,
elem67      number(15,6) NOT NULL,
elem77      number(15,6) NOT NULL);
REM
REM Adding constraints for table NOISEMATRIX_DATA
REM
alter table NOISEMATRIX_DATA add (
   constraint noisematrix_data_pk primary key (map_id,vec_index),
   constraint noisematrix_data_fk foreign key (map_id)
                          references noisematrix_map(map_id));
