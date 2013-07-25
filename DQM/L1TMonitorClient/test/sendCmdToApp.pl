#!/usr/bin/perl
                                                                                                                             
die "Usage sendCmdToApp.pl host port cmdFile\n" if @ARGV != 3;
                                                                                                                             
$host    = $ARGV[0];
$port    = $ARGV[1];
$cmdFile = $ARGV[2];

                                                                                                                             
die "sendCmdToApp.pl: The file \"$cmdFile\" does not exist\n" unless -e $cmdFile;
#abort is 3rd argument does not exist and print messege.


$curlCmd = "curl --stderr /dev/null -H \"Content-Type: text/xml\" -H \"Content-Description: SOAP Message\" -H \"SOAPAction: urn:xdaq-application:lid=0\" http://$host:$port -d \@$cmdFile";  
#send data in L1TClient.xml to server http://$host:$port, send standard error output to /dev/null, add Headers
# can look at example here: http://www.intertwingly.net/stories/2002/01/25/whatObjectDoesSoapAccess.html

print ">>>>>>>>>sendCmdToApp.pl: $curlCmd\n";                                                                                                                           

open CURL, "$curlCmd|";
# send child process (the curl command) that comunicate with perl until the process is over (pag201).
                                                                                                                             
print ">>>>>>>>>sendCmdToApp.pl: Sending command to executive $host:$port \n";
                                                                                                                             
while(<CURL>) {
#Until the proces is active, assign the job output to the variable $soapReply
    chomp;
    $soapReply .= $_;
    print ">>>>>>>>>sendCmdToApp.pl: $soapReply\n";
}
                                                                                                                             
if($soapReply =~ m/Response/) {
    print ">>>>>>>>>sendCmdToApp.pl: OK\n";
    exit 0;
} elsif($soapReply =~ m/Fault/) {
    print "sendCmdToApp.pl: FAULT\n";
    print "$soapReply\n";
    exit 1;
} elsif($soapReply eq "") {
    print "sendCmdToApp.pl: NONE\n";
    exit 1;
} else {
    print "sendCmdToApp.pl: UNKNOWN response\n";
    print "$soapReply\n";
    exit 1;
}
