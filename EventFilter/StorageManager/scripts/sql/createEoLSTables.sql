-- History, discussion and comments available on the wiki:
-- https://twiki.cern.ch/twiki/bin/viewauth/CMS/StorageManagerEndOfLumiHandling

create table CMS_STOMGR.RUNS (
  RUNNUMBER        NUMBER(10)     not NULL,
  INSTANCE         NUMBER(5)      not NULL,
  HOSTNAME         VARCHAR2(100),
  N_INSTANCES      NUMBER(5),
  N_LUMISECTIONS   NUMBER(10),
  MAX_LUMISECTION  NUMBER(10),
  LAST_CONSECUTIVE NUMBER(10),
  STATUS           NUMBER(1)      not NULL check (STATUS in (0, 1)),
  START_TIME       DATE           default sysdate not NULL,
  END_TIME         DATE           default NULL,
  constraint PK_RUNS primary key (RUNNUMBER, INSTANCE)
)
organization index
/
grant SELECT,INSERT,UPDATE on RUNS to CMS_STOMGR_W;

create table CMS_STOMGR.STREAMS (
  RUNNUMBER    NUMBER(10)    not NULL,
  LUMISECTION  NUMBER(10)    not NULL,
  STREAM       VARCHAR2(100) not NULL,
  INSTANCE     NUMBER(5)     not NULL,
  EOLS         NUMBER(1)     check (EOLS in (0, 1)) default 0,
  FILECOUNT    NUMBER(20),
  CTIME        DATE          default sysdate not NULL,
  constraint PK_STREAMS primary key (RUNNUMBER, STREAM, LUMISECTION, INSTANCE)
)
organization index
/
create index STREAMS_RUN_INSTANCE on CMS_STOMGR.STREAMS("RUNNUMBER","INSTANCE");
grant SELECT,INSERT on STREAMS to CMS_STOMGR_W;

