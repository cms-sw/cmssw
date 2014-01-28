die "Usage webPingXDAQ.pl host port nbRetries\n" if @ARGV != 3;

$host=$ARGV[0];
$port=$ARGV[1];
$nbRetries=$ARGV[2];

sub tryWebPing {
  my($host,$port) = @_;
  my $cmd = "wget -o /dev/null -O /dev/null http://${host}:${port}";
  my $cmdExitStatus = system($cmd);

  $cmdExitStatus;
}

print "Checking ${host}:${port} is listening";

for($i=1; $i<=$nbRetries; $i++) {
  print " .";
  $cmdExitStatus = &tryWebPing($host, $port);

  if($cmdExitStatus == 0)
  {
    print "\n${host}:${port} is listening\n";
    exit 0;
  }

  sleep 1;
}

print "\n${host}:${port} is NOT listening\n";
exit 1;

