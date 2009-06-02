<?php

$logfile = $_REQUEST['filename'];
?>

<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
<html>
<head>
<title>show log file</title>
</head>
<body>
<p>

<?php
unset( $lines );
if( file_exists( "$logfile" ) == "TRUE" ) {
  $fp = fopen( "$logfile", "r" );
  while( $line = fgets( $fp, 8192 )) {
    echo "$line <br>\n";
  }
  fclose( $fp ); 
}
else if( file_exists( "$logfile.gz" ) == "TRUE" ) {
  $fp = gzopen( "$logfile.gz", "r" );
  while( $line = gzgets( $fp, 8192 )) {
    echo "$line <br>\n";
  }
  gzclose( $fp );
}
?>  

</body>
</html>
