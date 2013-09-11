<?php

function safe($str){

$str = str_replace('"', "&#34", $str);
$str = str_replace("'", "&#39", $str);
$str = str_replace("<", "&lt;", $str);
$str = str_replace(">", "&gt;", $str);
$str = str_replace("\n", "", $str);
$str = str_replace("\r", "", $str);


return $str;
}

try
  { 
    $db = new PDO('sqlite:aq.db');

  $url="";

switch($_GET['action']){

case "newAuthor":

	$fullname = safe($_POST['fullname']);
	$email = safe($_POST['email']);
	$cadi_url = safe($_POST['cadi_url']);

	if (($fullname != "") && ($email != "")){
		$db->query("INSERT INTO author (id,fullname,email,cadi_url) VALUES (NULL,'".$fullname."','".$email."','".$cadi_url."')");
		$url = "?page=author&id=".$db->lastInsertId();
	}
	
break;

case "editAuthor":
	
	$author_id = safe($_POST['author_id']);
	$fullname = safe($_POST['fullname']);
	$email = safe($_POST['email']);
	$cadi_url = safe($_POST['cadi_url']);

	$return = safe($_POST['return']);
        $rid = safe($_POST['rid']);

	if (($fullname != "") && ($email != "") && is_numeric($author_id)){
		$db->query("UPDATE author SET fullname='".$fullname."', email='".$email."', cadi_url='".$cadi_url."' WHERE id=".$author_id);
		$url = "?page=".$return."&id=".$rid;
	}
break;

case "newInstitute":

	$name = safe($_POST['name']);
	$cadi_url = safe($_POST['cadi_url']);	

	if (($name != "")){
		$db->query("INSERT INTO institute (id,name,cadi_url) VALUES (NULL,'".$name."','".$cadi_url."')");
		$url = "?page=institute&id=".$db->lastInsertId();
	}
break;

case "editInstitute":
	
	$institute_id = safe($_POST['institute_id']);
	$name = safe($_POST['name']);
	$cadi_url = safe($_POST['cadi_url']);

	$return = safe($_POST['return']);
        $rid = safe($_POST['rid']);

	if (($name != "") && is_numeric($institute_id)){
		$db->query("UPDATE institute SET name='".$name."', cadi_url='".$cadi_url."' WHERE id=".$institute_id);
		$url = "?page=".$return."&id=".$rid;
	}
break;


case "addAuthorInstitute":

	$institute_id = safe($_POST['institute_id']);
	$author_id = safe($_POST['author_id']);
        $return = safe($_POST['return']);
	$rid = safe($_POST['rid']);

	if (is_numeric($institute_id) && is_numeric($author_id) && is_numeric($rid)){
		$db->query("INSERT INTO author_institute (id,institute_id,author_id) VALUES (NULL,'".$institute_id."','".$author_id."')");
		$url = "?page=".$return."&id=".$rid;			
	}


break;

case "removeAuthorInstitute":

	$ai_id = safe($_GET['ai_id']);
	$rid = safe($_GET['rid']);
	$return = safe($_GET['return']);

	if (is_numeric($ai_id) && is_numeric($rid)){
		$db->query("DELETE FROM author_institute WHERE id='".$ai_id."'");
		$url = "?page=".$return."&id=".$rid;				
	}
break;

case "newPublication":

	$title = safe($_POST['title']);
	$code = safe($_POST['code']);
	$date = safe($_POST['date_created']);
	$cadi_url = safe($_POST['cadi_url']);

	if (($title != "") && ($code != "") && ($date != "")){
		$db->query("INSERT INTO publication (id,title,code, date_created,cadi_url) VALUES (NULL,'".$title."','".$code."','".$date."','".$cadi_url."')");
		$url = "?page=publication&id=".$db->lastInsertId();
	}
break;

case "editPublication":
	
	$publication_id = safe($_POST['publication_id']);
	$title = safe($_POST['title']);
	$code = safe($_POST['code']);
	$date = safe($_POST['date_created']);
	$cadi_url = safe($_POST['cadi_url']);

	$return = safe($_POST['return']);
        $rid = safe($_POST['rid']);

	if (($title != "") && ($code != "") && ($date != "") && is_numeric($publication_id)){
		$db->query("UPDATE publication SET title='".$title."', code='".$code."', date_created='".$date."', cadi_url='".$cadi_url."' WHERE id=".$publication_id);
		$url = "?page=".$return."&id=".$rid;
	}
break;

case "addPublicationAuthor":
	$publication_id = $_POST['publication_id'];
	$author_id = $_POST['author_id'];
	
	$return = safe($_POST['return']);
        $rid = safe($_POST['rid']);

	if (is_numeric($publication_id) && is_numeric($author_id) && is_numeric($rid)){
		$db->query("INSERT INTO publication_author (id,publication_id,author_id) VALUES (NULL,'".$publication_id."','".$author_id."')");
		$url = "?page=".$return."&id=".$rid;			
	}
break;

case "removePublicationAuthor":
	$pa_id = safe($_GET['pa_id']);
	$rid = safe($_GET['rid']);
	$return = safe($_GET['return']);

	if (is_numeric($pa_id) && is_numeric($rid)){
		$db->query("DELETE FROM publication_author WHERE id='".$pa_id."'");
		$url = "?page=".$return."&id=".$rid;				
	}
	
break;

}

}
  catch(PDOException $e)
  {
    print 'Exception : '.$e->getMessage();
  }


?>

<script type="text/javascript">
window.location = "http://cmsdoxy.web.cern.ch/cmsdoxy/pai/<?php echo $url; ?>";
</script>