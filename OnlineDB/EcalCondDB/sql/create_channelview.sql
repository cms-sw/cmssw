/* Create channel mapping tables */

CREATE TABLE channelView (
  name VARCHAR2(32) NOT NULL,
  id1 NUMBER(10) DEFAULT NULL,
  id2 NUMBER(10) DEFAULT NULL,
  id3 NUMBER(10) DEFAULT NULL,
  maps_to VARCHAR2(32) NOT NULL,
  logic_id NUMBER(10) NOT NULL
);

ALTER TABLE channelView ADD CONSTRAINT cv_ix1 UNIQUE(name, id1, id2, id3, logic_id);
CREATE INDEX cv_ix2 ON channelView (maps_to);
CREATE INDEX cv_ix3 ON channelView (logic_id);



CREATE TABLE viewDescription (
  name VARCHAR2(32) NOT NULL,
  description VARCHAR2(4000),
  id1name VARCHAR2(32) DEFAULT NULL,
  id2name VARCHAR2(32) DEFAULT NULL,
  id3name VARCHAR2(32) DEFAULT NULL
);

ALTER TABLE viewDescription ADD CONSTRAINT cvd_pk PRIMARY KEY (name);
