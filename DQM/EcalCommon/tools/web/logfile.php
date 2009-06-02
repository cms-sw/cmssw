<?php

$run = $_REQUEST['run'];
$runtype = $_REQUEST['runtype'];

?>

<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>
<title>show log file</title>
</head>
<body>
<p>

<?php

$logfile1 = exec('ls /var/www/html/logs/*' . $run . '*.log | grep -v offset');
$logfile2 = exec('ls /var/www/html/logs/*' . $run . '*.log.gz | grep -v offset');

unset( $lines );
if( file_exists( "$logfile1" ) == "TRUE" ) {
  $fp = fopen( "$logfile1", "r" );
  while( $line = fgets( $fp, 8192 )) {
    echo "$line <br>\n";
  }
  fclose( $fp ); 
}
else if( file_exists( "$logfile2" ) == "TRUE" ) {
  $fp = gzopen( "$logfile2", "r" );
  while( $line = gzgets( $fp, 8192 )) {
    echo "$line <br>\n";
  }
  gzclose( $fp );
}
?>  

</body>
</html>
