CREATE TABLE st_ecalpedestals (
  iov_value_id NUMBER(10) NOT NULL,
  time NUMBER(38)
);

alter table st_ecalpedestals add constraint st_ecalpedestals_pk primary key (iov_value_id);


CREATE TABLE st_ecalpedestals_item (
  pos NUMBER(10) NOT NULL,
  iov_value_id NUMBER(10) NOT NULL,
  det_id NUMBER(10) NOT NULL,
  mean_x1 BINARY_FLOAT NOT NULL,
  mean_x12 BINARY_FLOAT NOT NULL,
  mean_x6 BINARY_FLOAT NOT NULL,
  rms_x1 BINARY_FLOAT NOT NULL,
  rms_x12 BINARY_FLOAT NOT NULL,
  rms_x6 BINARY_FLOAT NOT NULL
);

alter table st_ecalpedestals_item add constraint st_ecalpedestals_item_pk primary key (iov_value_id, pos);
alter table st_ecalpedestals_item add constraint st_ecalpedestals_item_fk foreign key (iov_value_id) references st_ecalpedestals (iov_value_id);
