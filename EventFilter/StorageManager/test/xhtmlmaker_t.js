function set_value( id, val )
{
  if( !document.getElementById ) return;
  document.getElementById(id).innerHTML = val;
}

function load_body()
{
  set_value( "my_id", "Not so fancy link" );
}

window.onload = load_body;
