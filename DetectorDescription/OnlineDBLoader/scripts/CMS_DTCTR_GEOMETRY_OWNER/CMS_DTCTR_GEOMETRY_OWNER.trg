-- D:\CMS-PROJECTS\Detector_Geometry_DB\CMS_DTCTR_GEOMETRY_OWNER.trg
--
-- Generated for Oracle 9i on Mon May 16  16:13:16 2005 by Server Generator 9.0.2.92.10
 

















































PROMPT Creating Trigger 'ZSEC_SEQ_TRIG'
CREATE OR REPLACE TRIGGER ZSEC_SEQ_TRIG
 BEFORE INSERT
 ON ZSECTIONS
 FOR EACH ROW
DECLARE

temp_seq number(38);
begin
   select zsec_seq.nextval into temp_seq from dual;
   :new.zsection_pk:=temp_seq;
end;
/
SHOW ERROR

















