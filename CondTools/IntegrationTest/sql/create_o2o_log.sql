/* Stores some details about a payload_o2o() transfer */

CREATE TABLE o2o_log (
  object_name		VARCHAR2(100),
  last_id		NUMBER,
  num_transferred	NUMBER,
  transfer_start 	TIMESTAMP,
  transfer_duration	INTERVAL DAY TO SECOND
);
