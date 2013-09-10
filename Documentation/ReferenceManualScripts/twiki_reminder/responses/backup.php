<?php

$now = date("Y_m_d_G.i");

$file = 'responses.txt';
$newfile = 'responses/responses'.$now.'.txt';

if (!copy($file, $newfile)) {
    echo "failed to copy $file...\n";
}
else{
    echo($newfile." Copied");
}
?>
