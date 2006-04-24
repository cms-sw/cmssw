CREATE TABLE o2o_setup (
  object_name VARCHAR2(32),
  top_level_table VARCHAR2(32),
  db_link VARCHAR2(32),
  PRIMARY KEY (object_name)
);

INSERT INTO o2o_setup VALUES ('EcalPedestals', 'ECALPEDESTALS', 'orcon.cern.ch@cms_cond_ecal');

COMMIT;
