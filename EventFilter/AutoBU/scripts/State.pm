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
my $RBid = 26;
my $EPid = 27;
my $SMid = 44;


# url's of different applications
# the current application state is reflected here
my $BU = "http://$host:$BUPort/urn:xdaq-application:lid=$BUid/";
my $RB = "http://$host:$RBPort/urn:xdaq-application:lid=$RBid/";
my $EP = "http://$host:$EPPort/urn:xdaq-application:lid=$EPid/updater";
my $SM = "http://$host:$SMPort/urn:xdaq-application:lid=$SMid/";


sub is_BU_initialised {
 my $response = `curl -s $BU | grep stateName`;
 if ($response =~ m/Halted/i) {
	return 1;
 }
 else {
	return 0;
 }
}

sub is_BU_ready {
 my $response = `curl -s $BU | grep stateName`;
 if ($response =~ m/Ready/i) {
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


1;
