webpackHotUpdate_N_E("pages/index",{

/***/ "./components/Nav.tsx":
/*!****************************!*\
  !*** ./components/Nav.tsx ***!
  \****************************/
/*! exports provided: Nav, default */
/***/ (function(module, __webpack_exports__, __webpack_require__) {

"use strict";
__webpack_require__.r(__webpack_exports__);
/* WEBPACK VAR INJECTION */(function(module) {/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, "Nav", function() { return Nav; });
/* harmony import */ var _babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(/*! @babel/runtime/helpers/esm/extends */ "./node_modules/@babel/runtime/helpers/esm/extends.js");
/* harmony import */ var _babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(/*! @babel/runtime/helpers/esm/slicedToArray */ "./node_modules/@babel/runtime/helpers/esm/slicedToArray.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(/*! react */ "./node_modules/react/index.js");
/* harmony import */ var react__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(react__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var antd__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(/*! antd */ "./node_modules/antd/es/index.js");
/* harmony import */ var _styledComponents__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(/*! ./styledComponents */ "./components/styledComponents.ts");
/* harmony import */ var _searchButton__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(/*! ./searchButton */ "./components/searchButton.tsx");
/* harmony import */ var _helpButton__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(/*! ./helpButton */ "./components/helpButton.tsx");
/* harmony import */ var _config_config__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(/*! ../config/config */ "./config/config.ts");



var _jsxFileName = "/mnt/c/Users/ernes/Desktop/cernProject/dqmgui_frontend/components/Nav.tsx",
    _this = undefined,
    _s = $RefreshSig$();

var __jsx = react__WEBPACK_IMPORTED_MODULE_2___default.a.createElement;






var Nav = function Nav(_ref) {
  _s();

  var initial_search_run_number = _ref.initial_search_run_number,
      initial_search_dataset_name = _ref.initial_search_dataset_name,
      setRunNumber = _ref.setRunNumber,
      setDatasetName = _ref.setDatasetName,
      handler = _ref.handler,
      type = _ref.type,
      defaultRunNumber = _ref.defaultRunNumber,
      defaultDatasetName = _ref.defaultDatasetName;

  var _Form$useForm = antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm(),
      _Form$useForm2 = Object(_babel_runtime_helpers_esm_slicedToArray__WEBPACK_IMPORTED_MODULE_1__["default"])(_Form$useForm, 1),
      form = _Form$useForm2[0];

  var _useState = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_run_number || ''),
      form_search_run_number = _useState[0],
      setFormRunNumber = _useState[1];

  var _useState2 = Object(react__WEBPACK_IMPORTED_MODULE_2__["useState"])(initial_search_dataset_name || ''),
      form_search_dataset_name = _useState2[0],
      setFormDatasetName = _useState2[1]; // We have to wait for changin initial_search_run_number and initial_search_dataset_name coming from query, because the first render they are undefined and therefore the initialValues doesn't grab them


  Object(react__WEBPACK_IMPORTED_MODULE_2__["useEffect"])(function () {
    form.resetFields();
    setFormRunNumber(initial_search_run_number || '');
    setFormDatasetName(initial_search_dataset_name || '');
  }, [initial_search_run_number, initial_search_dataset_name, form]);
  var layout = {
    labelCol: {
      span: 8
    },
    wrapperCol: {
      span: 16
    }
  };
  return __jsx("div", {
    style: {
      justifyContent: 'center'
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 52,
      columnNumber: 5
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["CustomForm"], Object(_babel_runtime_helpers_esm_extends__WEBPACK_IMPORTED_MODULE_0__["default"])({
    form: form,
    layout: 'inline',
    justifycontent: "center"
  }, layout, {
    name: "search_form".concat(type),
    className: "fieldLabel",
    initialValues: {
      run_number: initial_search_run_number,
      dataset_name: initial_search_dataset_name
    },
    onFinish: function onFinish() {
      setRunNumber && setRunNumber(form_search_run_number);
      setDatasetName && setDatasetName(form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 53,
      columnNumber: 7
    }
  }), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 69,
      columnNumber: 9
    }
  }, __jsx(_helpButton__WEBPACK_IMPORTED_MODULE_6__["QuestionButton"], {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 70,
      columnNumber: 11
    }
  })), __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "run_number",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 72,
      columnNumber: 9
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "run_number",
    onChange: function onChange(e) {
      return setFormRunNumber(e.target.value);
    },
    placeholder: "Enter run number",
    type: "text",
    name: "run_number",
    value: defaultRunNumber,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 73,
      columnNumber: 11
    }
  })), _config_config__WEBPACK_IMPORTED_MODULE_7__["functions_config"].mode !== 'ONLINE' && __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledFormItem"], {
    name: "dataset_name",
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 85,
      columnNumber: 11
    }
  }, __jsx(_styledComponents__WEBPACK_IMPORTED_MODULE_4__["StyledInput"], {
    id: "dataset_name",
    placeholder: "Enter dataset name",
    onChange: function onChange(e) {
      return setFormDatasetName(e.target.value);
    },
    type: "text",
    value: defaultDatasetName,
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 86,
      columnNumber: 13
    }
  })), __jsx(antd__WEBPACK_IMPORTED_MODULE_3__["Form"].Item, {
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 97,
      columnNumber: 9
    }
  }, __jsx(_searchButton__WEBPACK_IMPORTED_MODULE_5__["SearchButton"], {
    onClick: function onClick() {
      return handler(form_search_run_number, form_search_dataset_name);
    },
    __self: _this,
    __source: {
      fileName: _jsxFileName,
      lineNumber: 98,
      columnNumber: 11
    }
  }))));
};

_s(Nav, "d/o1hn25bH6EF0LAvbTEx8d/DOY=", false, function () {
  return [antd__WEBPACK_IMPORTED_MODULE_3__["Form"].useForm];
});

_c = Nav;
/* harmony default export */ __webpack_exports__["default"] = (Nav);

var _c;

$RefreshReg$(_c, "Nav");

;
    var _a, _b;
    // Legacy CSS implementations will `eval` browser code in a Node.js context
    // to extract CSS. For backwards compatibility, we need to check we're in a
    // browser context before continuing.
    if (typeof self !== 'undefined' &&
        // AMP / No-JS mode does not inject these helpers:
        '$RefreshHelpers$' in self) {
        var currentExports = module.__proto__.exports;
        var prevExports = (_b = (_a = module.hot.data) === null || _a === void 0 ? void 0 : _a.prevExports) !== null && _b !== void 0 ? _b : null;
        // This cannot happen in MainTemplate because the exports mismatch between
        // templating and execution.
        self.$RefreshHelpers$.registerExportsForReactRefresh(currentExports, module.i);
        // A module can be accepted automatically based on its exports, e.g. when
        // it is a Refresh Boundary.
        if (self.$RefreshHelpers$.isReactRefreshBoundary(currentExports)) {
            // Save the previous exports on update so we can compare the boundary
            // signatures.
            module.hot.dispose(function (data) {
                data.prevExports = currentExports;
            });
            // Unconditionally accept an update to this module, we'll check if it's
            // still a Refresh Boundary later.
            module.hot.accept();
            // This field is set when the previous version of this module was a
            // Refresh Boundary, letting us know we need to check for invalidation or
            // enqueue an update.
            if (prevExports !== null) {
                // A boundary can become ineligible if its exports are incompatible
                // with the previous exports.
                //
                // For example, if you add/remove/change exports, we'll want to
                // re-execute the importing modules, and force those components to
                // re-render. Similarly, if you convert a class component to a
                // function, we want to invalidate the boundary.
                if (self.$RefreshHelpers$.shouldInvalidateReactRefreshBoundary(prevExports, currentExports)) {
                    module.hot.invalidate();
                }
                else {
                    self.$RefreshHelpers$.scheduleUpdate();
                }
            }
        }
        else {
            // Since we just executed the code for the module, it's possible that the
            // new exports made it ineligible for being a boundary.
            // We only care about the case when we were _previously_ a boundary,
            // because we already accepted this update (accidental side effect).
            var isNoLongerABoundary = prevExports !== null;
            if (isNoLongerABoundary) {
                module.hot.invalidate();
            }
        }
    }

/* WEBPACK VAR INJECTION */}.call(this, __webpack_require__(/*! ./../node_modules/webpack/buildin/harmony-module.js */ "./node_modules/webpack/buildin/harmony-module.js")(module)))

/***/ })

})
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly9fTl9FLy4vY29tcG9uZW50cy9OYXYudHN4Il0sIm5hbWVzIjpbIk5hdiIsImluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIiLCJpbml0aWFsX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRSdW5OdW1iZXIiLCJzZXREYXRhc2V0TmFtZSIsImhhbmRsZXIiLCJ0eXBlIiwiZGVmYXVsdFJ1bk51bWJlciIsImRlZmF1bHREYXRhc2V0TmFtZSIsIkZvcm0iLCJ1c2VGb3JtIiwiZm9ybSIsInVzZVN0YXRlIiwiZm9ybV9zZWFyY2hfcnVuX251bWJlciIsInNldEZvcm1SdW5OdW1iZXIiLCJmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUiLCJzZXRGb3JtRGF0YXNldE5hbWUiLCJ1c2VFZmZlY3QiLCJyZXNldEZpZWxkcyIsImxheW91dCIsImxhYmVsQ29sIiwic3BhbiIsIndyYXBwZXJDb2wiLCJqdXN0aWZ5Q29udGVudCIsInJ1bl9udW1iZXIiLCJkYXRhc2V0X25hbWUiLCJlIiwidGFyZ2V0IiwidmFsdWUiLCJmdW5jdGlvbnNfY29uZmlnIiwibW9kZSJdLCJtYXBwaW5ncyI6Ijs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7QUFBQTtBQUNBO0FBRUE7QUFDQTtBQUNBO0FBQ0E7QUFjTyxJQUFNQSxHQUFHLEdBQUcsU0FBTkEsR0FBTSxPQVNIO0FBQUE7O0FBQUEsTUFSZEMseUJBUWMsUUFSZEEseUJBUWM7QUFBQSxNQVBkQywyQkFPYyxRQVBkQSwyQkFPYztBQUFBLE1BTmRDLFlBTWMsUUFOZEEsWUFNYztBQUFBLE1BTGRDLGNBS2MsUUFMZEEsY0FLYztBQUFBLE1BSmRDLE9BSWMsUUFKZEEsT0FJYztBQUFBLE1BSGRDLElBR2MsUUFIZEEsSUFHYztBQUFBLE1BRmRDLGdCQUVjLFFBRmRBLGdCQUVjO0FBQUEsTUFEZEMsa0JBQ2MsUUFEZEEsa0JBQ2M7O0FBQUEsc0JBQ0NDLHlDQUFJLENBQUNDLE9BQUwsRUFERDtBQUFBO0FBQUEsTUFDUEMs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QUFBO0FBQUEsS0FDRSxNQUFDLDBEQUFEO0FBQ0UsV0FBTyxFQUFFO0FBQUEsYUFDUEgsT0FBTyxDQUFDUSxzQkFBRCxFQUF5QkUsd0JBQXpCLENBREE7QUFBQSxLQURYO0FBQUE7QUFBQTtBQUFBO0FBQUE7QUFBQTtBQUFBO0FBQUEsSUFERixDQTVDRixDQURGLENBREY7QUF3REQsQ0F0Rk07O0dBQU1mLEc7VUFVSVMseUNBQUksQ0FBQ0MsTzs7O0tBVlRWLEc7QUF3RkVBLGtFQUFmIiwiZmlsZSI6InN0YXRpYy93ZWJwYWNrL3BhZ2VzL2luZGV4LjA5Yzk5ZjVhOWFmZjVkNDkyMGM3LmhvdC11cGRhdGUuanMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgUmVhY3QsIHsgQ2hhbmdlRXZlbnQsIERpc3BhdGNoLCB1c2VFZmZlY3QsIHVzZVN0YXRlIH0gZnJvbSAncmVhY3QnO1xyXG5pbXBvcnQgeyBGb3JtIH0gZnJvbSAnYW50ZCc7XHJcblxyXG5pbXBvcnQgeyBTdHlsZWRGb3JtSXRlbSwgU3R5bGVkSW5wdXQsIEN1c3RvbUZvcm0gfSBmcm9tICcuL3N0eWxlZENvbXBvbmVudHMnO1xyXG5pbXBvcnQgeyBTZWFyY2hCdXR0b24gfSBmcm9tICcuL3NlYXJjaEJ1dHRvbic7XHJcbmltcG9ydCB7IFF1ZXN0aW9uQnV0dG9uIH0gZnJvbSAnLi9oZWxwQnV0dG9uJztcclxuaW1wb3J0IHsgZnVuY3Rpb25zX2NvbmZpZyB9IGZyb20gJy4uL2NvbmZpZy9jb25maWcnO1xyXG5cclxuaW50ZXJmYWNlIE5hdlByb3BzIHtcclxuICBzZXRSdW5OdW1iZXI/OiBEaXNwYXRjaDxhbnk+O1xyXG4gIHNldERhdGFzZXROYW1lPzogRGlzcGF0Y2g8YW55PjtcclxuICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyPzogc3RyaW5nO1xyXG4gIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZT86IHN0cmluZztcclxuICBpbml0aWFsX3NlYXJjaF9sdW1pc2VjdGlvbj86IHN0cmluZztcclxuICBoYW5kbGVyKHNlYXJjaF9ieV9ydW5fbnVtYmVyOiBzdHJpbmcsIHNlYXJjaF9ieV9kYXRhc2V0X25hbWU6IHN0cmluZyk6IHZvaWQ7XHJcbiAgdHlwZTogc3RyaW5nO1xyXG4gIGRlZmF1bHRSdW5OdW1iZXI/OiB1bmRlZmluZWQgfCBzdHJpbmc7XHJcbiAgZGVmYXVsdERhdGFzZXROYW1lPzogc3RyaW5nIHwgdW5kZWZpbmVkO1xyXG59XHJcblxyXG5leHBvcnQgY29uc3QgTmF2ID0gKHtcclxuICBpbml0aWFsX3NlYXJjaF9ydW5fbnVtYmVyLFxyXG4gIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSxcclxuICBzZXRSdW5OdW1iZXIsXHJcbiAgc2V0RGF0YXNldE5hbWUsXHJcbiAgaGFuZGxlcixcclxuICB0eXBlLFxyXG4gIGRlZmF1bHRSdW5OdW1iZXIsXHJcbiAgZGVmYXVsdERhdGFzZXROYW1lLFxyXG59OiBOYXZQcm9wcykgPT4ge1xyXG4gIGNvbnN0IFtmb3JtXSA9IEZvcm0udXNlRm9ybSgpO1xyXG4gIGNvbnN0IFtmb3JtX3NlYXJjaF9ydW5fbnVtYmVyLCBzZXRGb3JtUnVuTnVtYmVyXSA9IHVzZVN0YXRlKFxyXG4gICAgaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciB8fCAnJ1xyXG4gICk7XHJcbiAgY29uc3QgW2Zvcm1fc2VhcmNoX2RhdGFzZXRfbmFtZSwgc2V0Rm9ybURhdGFzZXROYW1lXSA9IHVzZVN0YXRlKFxyXG4gICAgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lIHx8ICcnXHJcbiAgKTtcclxuXHJcbiAgLy8gV2UgaGF2ZSB0byB3YWl0IGZvciBjaGFuZ2luIGluaXRpYWxfc2VhcmNoX3J1bl9udW1iZXIgYW5kIGluaXRpYWxfc2VhcmNoX2RhdGFzZXRfbmFtZSBjb21pbmcgZnJvbSBxdWVyeSwgYmVjYXVzZSB0aGUgZmlyc3QgcmVuZGVyIHRoZXkgYXJlIHVuZGVmaW5lZCBhbmQgdGhlcmVmb3JlIHRoZSBpbml0aWFsVmFsdWVzIGRvZXNuJ3QgZ3JhYiB0aGVtXHJcbiAgdXNlRWZmZWN0KCgpID0+IHtcclxuICAgIGZvcm0ucmVzZXRGaWVsZHMoKTtcclxuICAgIHNldEZvcm1SdW5OdW1iZXIoaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciB8fCAnJyk7XHJcbiAgICBzZXRGb3JtRGF0YXNldE5hbWUoaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lIHx8ICcnKTtcclxuICB9LCBbaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlciwgaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lLCBmb3JtXSk7XHJcblxyXG4gIGNvbnN0IGxheW91dCA9IHtcclxuICAgIGxhYmVsQ29sOiB7IHNwYW46IDggfSxcclxuICAgIHdyYXBwZXJDb2w6IHsgc3BhbjogMTYgfSxcclxuICB9O1xyXG5cclxuICByZXR1cm4gKFxyXG4gICAgPGRpdiBzdHlsZT17e2p1c3RpZnlDb250ZW50OiAnY2VudGVyJ319PiBcclxuICAgICAgPEN1c3RvbUZvcm1cclxuICAgICAgICBmb3JtPXtmb3JtfVxyXG4gICAgICAgIGxheW91dD17J2lubGluZSd9XHJcbiAgICAgICAganVzdGlmeWNvbnRlbnQ9XCJjZW50ZXJcIlxyXG4gICAgICAgIHsuLi5sYXlvdXR9XHJcbiAgICAgICAgbmFtZT17YHNlYXJjaF9mb3JtJHt0eXBlfWB9XHJcbiAgICAgICAgY2xhc3NOYW1lPVwiZmllbGRMYWJlbFwiXHJcbiAgICAgICAgaW5pdGlhbFZhbHVlcz17e1xyXG4gICAgICAgICAgcnVuX251bWJlcjogaW5pdGlhbF9zZWFyY2hfcnVuX251bWJlcixcclxuICAgICAgICAgIGRhdGFzZXRfbmFtZTogaW5pdGlhbF9zZWFyY2hfZGF0YXNldF9uYW1lLFxyXG4gICAgICAgIH19XHJcbiAgICAgICAgb25GaW5pc2g9eygpID0+IHtcclxuICAgICAgICAgIHNldFJ1bk51bWJlciAmJiBzZXRSdW5OdW1iZXIoZm9ybV9zZWFyY2hfcnVuX251bWJlcik7XHJcbiAgICAgICAgICBzZXREYXRhc2V0TmFtZSAmJiBzZXREYXRhc2V0TmFtZShmb3JtX3NlYXJjaF9kYXRhc2V0X25hbWUpO1xyXG4gICAgICAgIH19XHJcbiAgICAgID5cclxuICAgICAgICA8Rm9ybS5JdGVtPlxyXG4gICAgICAgICAgPFF1ZXN0aW9uQnV0dG9uIC8+XHJcbiAgICAgICAgPC9Gb3JtLkl0ZW0+XHJcbiAgICAgICAgPFN0eWxlZEZvcm1JdGVtIG5hbWU9XCJydW5fbnVtYmVyXCI+XHJcbiAgICAgICAgICA8U3R5bGVkSW5wdXRcclxuICAgICAgICAgICAgaWQ9XCJydW5fbnVtYmVyXCJcclxuICAgICAgICAgICAgb25DaGFuZ2U9eyhlOiBDaGFuZ2VFdmVudDxIVE1MSW5wdXRFbGVtZW50PikgPT5cclxuICAgICAgICAgICAgICBzZXRGb3JtUnVuTnVtYmVyKGUudGFyZ2V0LnZhbHVlKVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAgIHBsYWNlaG9sZGVyPVwiRW50ZXIgcnVuIG51bWJlclwiXHJcbiAgICAgICAgICAgIHR5cGU9XCJ0ZXh0XCJcclxuICAgICAgICAgICAgbmFtZT1cInJ1bl9udW1iZXJcIlxyXG4gICAgICAgICAgICB2YWx1ZT17ZGVmYXVsdFJ1bk51bWJlcn1cclxuICAgICAgICAgIC8+XHJcbiAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cclxuICAgICAgICB7ZnVuY3Rpb25zX2NvbmZpZy5tb2RlICE9PSAnT05MSU5FJyAmJiAoXHJcbiAgICAgICAgICA8U3R5bGVkRm9ybUl0ZW0gbmFtZT1cImRhdGFzZXRfbmFtZVwiPlxyXG4gICAgICAgICAgICA8U3R5bGVkSW5wdXRcclxuICAgICAgICAgICAgICBpZD1cImRhdGFzZXRfbmFtZVwiXHJcbiAgICAgICAgICAgICAgcGxhY2Vob2xkZXI9XCJFbnRlciBkYXRhc2V0IG5hbWVcIlxyXG4gICAgICAgICAgICAgIG9uQ2hhbmdlPXsoZTogQ2hhbmdlRXZlbnQ8SFRNTElucHV0RWxlbWVudD4pID0+XHJcbiAgICAgICAgICAgICAgICBzZXRGb3JtRGF0YXNldE5hbWUoZS50YXJnZXQudmFsdWUpXHJcbiAgICAgICAgICAgICAgfVxyXG4gICAgICAgICAgICAgIHR5cGU9XCJ0ZXh0XCJcclxuICAgICAgICAgICAgICB2YWx1ZT17ZGVmYXVsdERhdGFzZXROYW1lfVxyXG4gICAgICAgICAgICAvPlxyXG4gICAgICAgICAgPC9TdHlsZWRGb3JtSXRlbT5cclxuICAgICAgICApfVxyXG4gICAgICAgIDxGb3JtLkl0ZW0gPlxyXG4gICAgICAgICAgPFNlYXJjaEJ1dHRvblxyXG4gICAgICAgICAgICBvbkNsaWNrPXsoKSA9PlxyXG4gICAgICAgICAgICAgIGhhbmRsZXIoZm9ybV9zZWFyY2hfcnVuX251bWJlciwgZm9ybV9zZWFyY2hfZGF0YXNldF9uYW1lKVxyXG4gICAgICAgICAgICB9XHJcbiAgICAgICAgICAvPlxyXG4gICAgICAgIDwvRm9ybS5JdGVtPlxyXG4gICAgICA8L0N1c3RvbUZvcm0+XHJcbiAgICA8L2Rpdj5cclxuICApO1xyXG59O1xyXG5cclxuZXhwb3J0IGRlZmF1bHQgTmF2O1xyXG4iXSwic291cmNlUm9vdCI6IiJ9