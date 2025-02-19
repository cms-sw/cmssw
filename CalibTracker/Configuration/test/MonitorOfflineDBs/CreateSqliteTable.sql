DROP TABLE DBtags;
CREATE TABLE DBtags(
db  VARCHAR2(200) NOT NULL,
account  VARCHAR2(200) NOT NULL,
tag   VARCHAR2(200) NOT NULL,
PRIMARY KEY(db, account, tag)
);

---INSERT INTO DBtags values('A' , 'A1');
---INSERT INTO DBtags values('A' , 'A2');
---INSERT INTO DBtags values('B' , 'B1');
---INSERT INTO DBtags values('C' , 'B1');
---INSERT INTO DBtags values('A' , 'A1');
