CREATE OR REPLACE PACKAGE fill_dummy IS
procedure fill_pedestals;
procedure fill_gains;
end;
/
show errors

create or replace package body fill_dummy is
norm number := 2147483648;
TYPE table_type IS TABLE OF number INDEX BY BINARY_INTEGER;
chambers table_type;

procedure fill_pedestals is
run number;
layer number;
i number;
j number;
k number;
i_record_id number;
i_map_id number;
i_map_index number;
mean_ped number;
mean_rms number;
rndm number;
i_ped number;
i_rms number;
BEGIN
select max(record_id) into i_record_id from pedestals;
if i_record_id is null then
 i_record_id:=0;
end if;
select max(map_id) into i_map_id from pedestals_map;
if i_map_id is null then
 i_map_id:=0;
end if;
for run in 1..5 loop
 i_record_id:=i_record_id+1;
 insert into pedestals values (i_record_id,run,sysdate,sysdate);
 i_map_index:=0;
 for i in 1..18 loop
  select dbms_random.random into rndm from dual;
  mean_ped:=597+rndm/norm*21;
  select dbms_random.random into rndm from dual;
  mean_rms:=2.07+rndm/norm*0.24;
  for j in 1..6 loop
   i_map_id:=i_map_id+1;
   i_map_index:=i_map_index+1;
   layer:=chambers(i)+j;
   insert into pedestals_map values (i_map_id,i_record_id,i_map_index,layer);
   for k in 1..80 loop
    select dbms_random.random into rndm from dual;
    i_ped:=mean_ped+rndm/norm*56;
    i_rms:=mean_rms+rndm/norm*0.35;
    insert into pedestals_data values(i_map_id,k,i_ped,i_rms);
   end loop;
  end loop;
 end loop;
end loop;
END fill_pedestals;

procedure fill_gains is
run number;
layer number;
i number;
j number;
k number;
i_record_id number;
i_map_id number;
i_map_index number;
mean_gain_slope number;
mean_gain_intercept number;
mean_gain_chi2 number;
rndm number;
i_gain_slope number;
i_gain_intercept number;
i_gain_chi2 number;
BEGIN
select max(record_id) into i_record_id from gains;
if i_record_id is null then
 i_record_id:=0;
end if;
select max(map_id) into i_map_id from gains_map;
if i_map_id is null then
 i_map_id:=0;
end if;
for run in 1..5 loop
 i_record_id:=i_record_id+1;
 insert into gains values (i_record_id,run,sysdate,sysdate);
 i_map_index:=0;
 for i in 1..18 loop
  select dbms_random.random into rndm from dual;
  mean_gain_slope:=7.31+rndm/norm*0.68;
  select dbms_random.random into rndm from dual;
  mean_gain_intercept:=-13.2+rndm/norm*8.8;
  select dbms_random.random into rndm from dual;
  mean_gain_chi2:=2.17+rndm/norm*0.76;
  for j in 1..6 loop
   i_map_id:=i_map_id+1;
   i_map_index:=i_map_index+1;
   layer:=chambers(i)+j;
--   DBMS_OUTPUT.PUT_LINE (i_map_id||' '||i_record_id||' '||i_map_index||
--    ' '||layer);
   insert into gains_map values (i_map_id,i_record_id,i_map_index,layer);
   for k in 1..80 loop
    select dbms_random.random into rndm from dual;
    i_gain_slope:=mean_gain_slope+rndm/norm*0.58;
    i_gain_intercept:=mean_gain_intercept+rndm/norm*11.8;
    i_gain_chi2:=mean_gain_chi2*(1+rndm/norm*0.8);
    insert into gains_data values(i_map_id,k,i_gain_slope,i_gain_intercept,
    i_gain_chi2);
   end loop;
  end loop;
 end loop;
end loop;
END fill_gains;

BEGIN
chambers(1):= 220121140;
chambers(2):= 220121150;
chambers(3):= 220121160;
chambers(4):= 220121270;
chambers(5):= 220121280;
chambers(6):= 220121290;
chambers(7):= 220121300;
chambers(8):= 220121310;
chambers(9):= 220121320;
chambers(10):= 220131140;
chambers(11):= 220131150;
chambers(12):= 220131160;
chambers(13):= 220131270;
chambers(14):= 220131280;
chambers(15):= 220131290;
chambers(16):= 220131300;
chambers(17):= 220131310;
chambers(18):= 220131320;
end;
/
show errors
