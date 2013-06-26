#!/usr/bin/env perl

$iov = 1;
$day = 1;
$month = 1;   

srand ($time ^ $$ ^ unpack "%L*", 'ps axww | gzip');

print "VARIABLE IOV_ID NUMBER;\nBEGIN\n";

%summary;

$summary{'A'} = '';
$summary{'B'} = '';
$summary{'C'} = '';

while ($iov < 100) {
    $i = int(rand(6))+ 1;
    $x = rand();
    if ($day > 28) {
	$month++;
	$day = 1;
    }
    $sql = "SELECT TEST_IOV_SQ.NextVal INTO :IOV_ID FROM DUAL;\n";
    print $sql;
    $sql = sprintf("INSERT INTO TEST_IOV VALUES(:iov_id, %d, 1, " .
		   "TO_DATE(\'%02d-%02d-2010 00:00:00\', \'DD-MM-YYYY HH24:MI:SS\'), " .
		   "TO_DATE(\'31-12-9999 23:59:59\', \'DD-MM-YYYY HH24:MI:SS\')) " .
		   ";", 
		   $i, $day, $month);
    print $sql . "\n";
    $sql = "";
    if ($i == 1) {
	$sql .= "INSERT INTO TEST_A VALUES (:iov_id, $x);\n";
	$summary{'A'} .= sprintf("%02d%02d\n", $day, $month);
    } elsif ($i == 2) {
	$sql .= "INSERT INTO TEST_B VALUES (:iov_id, $x);\n";
	$summary{'B'} .= sprintf("%02d%02d\n", $day, $month);
    } elsif ($i == 3) {
	$sql .= "INSERT INTO TEST_A VALUES (:iov_id, $x);\n";
	$sql .= "INSERT INTO TEST_B VALUES (:iov_id, $x);\n";
	$summary{'A'} .= sprintf("%02d%02d\n", $day, $month);
	$summary{'B'} .= sprintf("%02d%02d\n", $day, $month);
    } elsif ($i == 4) {
	$sql .= "INSERT INTO TEST_C VALUES (:iov_id, $x);\n";
	$summary{'C'} .= sprintf("%02d%02d\n", $day, $month);
    } elsif ($i == 5) {
	$sql .= "INSERT INTO TEST_A VALUES (:iov_id, $x);\n";
	$sql .= "INSERT INTO TEST_C VALUES (:iov_id, $x);\n";
	$summary{'A'} .= sprintf("%02d%02d\n", $day, $month);
	$summary{'C'} .= sprintf("%02d%02d\n", $day, $month);
    } elsif ($i == 6) {
	$sql .= "INSERT INTO TEST_B VALUES (:iov_id, $x);\n";
	$sql .= "INSERT INTO TEST_C VALUES (:iov_id, $x);\n";
	$summary{'B'} .= sprintf("%02d%02d\n", $day, $month);
	$summary{'C'} .= sprintf("%02d%02d\n", $day, $month);
    } else {
	$sql .= "INSERT INTO TEST_A VALUES (:iov_id, $x);\n";
	$sql .= "INSERT INTO TEST_B VALUES (:iov_id, $x);\n";
	$sql .= "INSERT INTO TEST_C VALUES (:iov_id, $x);\n";
	$summary{'A'} .= sprintf("%02d%02d\n", $day, $month);
	$summary{'B'} .= sprintf("%02d%02d\n", $day, $month);
	$summary{'C'} .= sprintf("%02d%02d\n", $day, $month);
    }
    print $sql . "COMMIT;\n";
    $iov++;
    $day++;
}
print "END;\n/\n";

open OUT, ">report.txt";
print OUT "A was measured on\n";
print OUT $summary{'A'};
print OUT "\nB was measured on\n";
print OUT $summary{'B'};
print OUT "\nC was measured on\n";
print OUT $summary{'C'};
close OUT;
