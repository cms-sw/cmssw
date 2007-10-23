Channelview Table

Overview

  The channelview contains mappings from array based numbering systems
  used by ECAL to a single integer (logic_id).  It also contains
  mappings from one type of channel to another, for example from
  crystal number to high voltage channel.

  TODO:  Describe the mappings in more detail.



Creating channelview table from scratch

  To create the channelview table from scratch, use the
  "define-channels.pl" script in the perl/ directory.  

  TODO:  more information about this



Copying channelview tables from one DB to another

  Because creating the channelview table from scratch is quite slow,
  you may wish to simply copy a complete channelview table from one
  table to another.  There are two ways to do this.



  Method 1 - Using a database link

  -- Connect to the target database
  lxplus>  sqlplus [targetDB]

  -- Create backups of the channelview tables
  SQL>  create table bkup_channelview as select * from channelview;
  SQL>  create table bkup_viewdescription as select * from
  channelview;

  -- Create the database link
  SQL>  create database link ecalh4db.world connect to cond01
  identified by [password] using 'pccmsecdb.cern.ch:1521/ecalh4db';
  
  -- Make sure the database link works
  SQL>  select count(*) from channelview@ecalh4db.world;
  
  -- Drop the old channelview tables
  SQL>  drop table channelview;
  SQL>  drop table viewdescription;

  -- Copy the other channelview tables using the DB link
  SQL>  create table channelview as select * from
  channelview@ecalh4db.world;
  SQL>  create table viewdescription as select * from
  viewdescription@ecalh4db.world;

  -- Remove the backup tables
  SQL>  drop table bkup_channelview;
  SQL>  drop table bkup_viewdescription;



  Method 2 - Using oracle tools EXP and IMP

  You can also use the Oracle export/import tools to do the table
  copy.  Note you may run into some problems with this method if the
  databases are of different version.  Use the setoraenv command to
  set your environment to use Oracle tools appropriate to your target
  database.

  #  Have a look at the help files
  lxplus>  exp help=yes
  lxplus>  imp help=yes

  # Export the tables
  lxplus>  exp [source connection string] TABLES=(CHANNELVIEW,VIEWDESCRIPTION)

  # Import the tables
  lxplus>  imp [dest connection string] TABLES=(CHANNELVIEW,VIEWDESCRIPTION)
