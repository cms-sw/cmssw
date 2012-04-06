package State;

# ***********************
# configurable parameters
# ***********************

my $host = `hostname`;
chop($host);

# application ports
my $BUPort = 40000;
my $RBPort = 40000;
my $EPPort = 40002;
my $SMPort = 40001;

# application id's
my $BUid = 21;
my $PBid = 22;
my $RBid = 26;
my $EPid = 27;
my $SMid = 44;
my $ATCPid = 47;

# url's of different applications
# the current application state is reflected here
my $BU = "http://$host:$BUPort/urn:xdaq-application:lid=$BUid/";
my $RB = "http://$host:$RBPort/urn:xdaq-application:lid=$RBid/";
my $PB = "http://$host:$BUPort/urn:xdaq-application:lid=$PBid/updater";
my $EP = "http://$host:$EPPort/urn:xdaq-application:lid=$EPid/updater";
my $SM = "http://$host:$SMPort/urn:xdaq-application:lid=$SMid";
my $ATCPPB = "http://$host:$BUPort/urn:xdaq-application:lid=$ATCPid/";
my $ATCPSM = "http://$host:$RBPort/urn:xdaq-application:lid=$ATCPid/";


sub is_BU_initialised {
 my $response = `curl -s $BU`;
 if ($response =~ m/Halted/i) {
	return 1;
 }
 else {
	return 0;
}
}

sub is_BU_ready {
 my $response = `curl -s $BU`;
 if ($response =~ m/Stopped/i || $response =~ m/Ready/i) {
	return 1;
 }
 else {
	return 0;
 }
}

sub is_BU_failed {
 my $response = `curl -s $BU | grep evf::BU`;
 if ($response =~ m/Failed/i) {
  return 1;
 }
 else {
  return 0;
 }
}

sub is_RB_initialised {
 my $response = `curl -s $RB | grep stateName`;
 if ($response =~ m/Halted/i) {
	return 1;
 }
 else {
	return 0;
 }
}

sub is_RB_ready {
 my $response = `curl -s $RB | grep stateName`;
 if ($response =~ m/Ready/i) {
	return 1;
 }
 else {
	return 0;
 }
}

sub is_RB_failed{
 my $response = `curl -s $RB | grep stateName`;
 if ($response =~ m/Failed/i) {
  return 1;
 }
 else {
  return 0;
 }
}

sub is_EP_initialised {
 my $response = `curl -s $EP`;
 if ($response =~ m/Halted/i) {
	return 1;
 }
 else {
	return 0;
 }
}

sub is_EP_ready {
 my $response = `curl -s $EP`;
 if ($response =~ m/Ready/i) {
	return 1;
 }
 else {
	return 0;
 }
}

sub is_EP_failed {
 my $response = `curl -s $EP`;
 if ($response =~ m/Failed/i) {
  return 1;
 }
 else {
  return 0;
 }
}


sub is_SM_initialised {
 my $response = `curl -s $SM`;
 if ($response =~ m/Halted/i) {
	return 1;
 }
 else {
	return 0;
 }
}

sub is_SM_ready {
 my $response = `curl -s $SM`;
 if ($response =~ m/Ready/i) {
	return 1;
 }
 else {
	return 0;
 }
}

sub is_SM_failed {
  my $response = `curl -s $SM`;
  if ($response =~ m/Failed/i) {
    return 1;
  }
  else {
    return 0;
  }
}

sub is_PB_initialised {
  my $response = `curl -s $PB`;
  if ($response =~ m/Halted/i) {
    return 1;
  }
  else {
    return 0;
  }
}

sub is_PB_ready {
 my $response = `curl -s $PB`;
 if ($response =~ m/Ready/i) {
  return 1;
 }
 else {
  return 0;
 }
}

sub is_PB_failed {
 my $response = `curl -s $PB`;
 if ($response =~ m/Failed/i) {
  return 1;
 }
 else {
  return 0;
 }
}

sub is_ATCPPB_initialised {
  my $response = `curl -s $ATCPPB | grep Halted`;
  if  ($response = ~m/Halted/i) {
    return 1;
  }
  else {
    return 0;
  }
}

sub is_ATCPSM_initialised {
  my $response = `curl -s $ATCPSM | grep Halted`;
  if  ($response = ~m/Halted/i) {
    return 1;
  }
  else {
    return 0;
  }
}

sub is_ATCPPB_ready {
  my $response = `curl -s $ATCPPB | grep Ready`;
  if  ($response = ~m/Ready/i) {
    return 1;
  }
  else {
    return 0;
  }
}

sub is_ATCPSM_ready {
  my $response = `curl -s $ATCPSM | grep Ready`;
  if  ($response = ~m/Ready/i) {
    return 1;
  }
  else {
    return 0;
  }
}

#always return false
sub is_ATCPPB_failed {
  return 0;
}

sub is_ATCPSM_failed {
  return 0;
}


1;
