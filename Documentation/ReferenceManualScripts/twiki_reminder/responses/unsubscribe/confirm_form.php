<?php
  $today = date('d-m-y H:m:s');
  $email = empty($_POST["email"]) ? "" : $_POST["email"];
  $reason = empty($_POST["reason"]) ? "" : $_POST["reason"];
  $other = empty($_POST["other"]) ? "" : $_POST["other"];

  if(!$email && (!$reason || !$other)){
    echo "Not all rows filled!";
  }
  else
  {
    $file = "form_fill.txt";
    $txt = fopen($file, "a");
    fwrite($txt,$today.", ");
    fwrite($txt,$email.", ");
    fwrite($txt,$reason.", ");
    fwrite($txt,$other."\r\n");
    fclose($txt);
    echo "Form successfully sent.";
  }
 
?>